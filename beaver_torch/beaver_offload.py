"""
beaver_offload.py — BeaverManager and BaselineManager for E2e Experiments.

Per e2e_experiment_plan.md:
  - BeaverManager: GPU → PM (COW) + async prefetch PM → DRAM
  - BaselineManager: GPU → DRAM → PM (sync+rename, crash-consistent)

Both managers implement the same interface:
  - evict_readonly(block_idx, params): release GPU memory, no PM write
  - evict_dirty(block_idx, params): write to PM + trigger prefetch
  - restore(block_idx, params): DRAM (hit) or PM (miss) → GPU
  - prefetch(block_idx): PM → DRAM (async, during gap period)
  - reset_dram_pool(): reclaim DRAM pool for next iteration

Training loop integration (per e2e_experiment_plan.md):
  Forward:  restore → compute → evict_readonly (params not modified)
  Backward: restore → compute → evict_readonly (params not modified)
  Optimizer step: restore → Adam update → evict_dirty (params modified)
"""

import os
import math
import time
import numpy as np
import torch

# Adam hyperparams (must match training script)
LR           = 3e-4
WEIGHT_DECAY = 0.01
BETA1, BETA2 = 0.9, 0.999
EPS          = 1e-8

PAGE_SIZE      = 4096
CHUNK_PAGES    = 1017           # F2FS_ADDRS_PER_INODE limit
CHUNK_BYTES    = CHUNK_PAGES * PAGE_SIZE   # ~4.16 MiB per file


def _import_beaver_ext():
    """Import beaver_ext; raise a clear error if not built yet."""
    try:
        import beaver_ext
        return beaver_ext
    except ImportError as e:
        raise ImportError(
            "beaver_ext not found. Build it first:\n"
            "  cd beaver_torch && pip install -e .\n"
            f"Original error: {e}"
        ) from e


class BeaverManager:
    """
    Per-layer parameter offload to Intel Optane PM via Beaver COW.

    Write path (evict_dirty):
      GPU tensor → Beaver COW → PM (crash-consistent)
      + trigger async prefetch: PM → pinned DRAM

    Read path (restore):
      DRAM (prefetch hit) → GPU (fast)
      PM (miss) → GPU (slow fallback)

    evict_readonly:
      Only release GPU memory, no PM write (forward/backward evict)
    """

    def __init__(self, blocks, lr=LR, weight_decay=WEIGHT_DECAY,
                 beta1=BETA1, beta2=BETA2, eps=EPS):
        self._ext = _import_beaver_ext()
        self.num_blocks  = len(blocks)
        self.lr          = lr
        self.wd          = weight_decay
        self.beta1       = beta1
        self.beta2       = beta2
        self.eps         = eps
        self.step_count  = 0

        self._write_ms = []
        self._read_ms  = []

        # ── Compute sizing ──────────────────────────────────────────────
        # fds[block_idx][param_idx] = list of chunk fds (int)
        # shapes[block_idx][param_idx] = (num_elements, dtype)
        self.fds    = []
        self.shapes = []

        total_files      = 0
        total_data_pages = 0

        for blk in blocks:
            blk_fds    = []
            blk_shapes = []
            for p in blk.parameters():
                n_bytes = p.data.nbytes
                n_pages = math.ceil(n_bytes / PAGE_SIZE)
                n_chunks = math.ceil(n_pages / CHUNK_PAGES)
                blk_fds.append([None] * n_chunks)       # fds filled in below
                blk_shapes.append((p.data.numel(), p.data.dtype))
                total_files      += n_chunks
                total_data_pages += n_pages
            self.fds.append(blk_fds)
            self.shapes.append(blk_shapes)

        # Add margin so FS doesn't run out on the boundary
        max_inodes      = total_files + 16
        max_data_pages  = total_data_pages + 256
        # DRAM pool must hold all pages simultaneously (prefetch all before read)
        dram_pool_pages = total_data_pages + 256

        print(f"BeaverManager sizing:")
        print(f"  blocks            : {self.num_blocks}")
        print(f"  total chunk files : {total_files}  (max_inodes={max_inodes})")
        print(f"  total data pages  : {total_data_pages} "
              f"({total_data_pages * PAGE_SIZE / 1024**3:.2f} GB)")
        print(f"  DRAM pool         : {dram_pool_pages} pages "
              f"({dram_pool_pages * PAGE_SIZE / 1024**3:.2f} GB pinned)")

        # ── Init FS ─────────────────────────────────────────────────────
        self._ext.init(max_inodes, max_data_pages, dram_pool_pages)

        # ── Pre-create all files ────────────────────────────────────────
        print("  Creating chunk files on PM FS ...")
        hash_counter = 0
        for bi, blk in enumerate(blocks):
            for pi, p in enumerate(blk.parameters()):
                n_bytes  = p.data.nbytes
                n_pages  = math.ceil(n_bytes / PAGE_SIZE)
                n_chunks = len(self.fds[bi][pi])
                for ci in range(n_chunks):
                    # Unique hash per file: blend block, param, chunk indices
                    h = (hash_counter * 7919 + 0xDEADBEEF) & 0xFFFFFFFF
                    hash_counter += 1
                    fd = self._ext.create_file(h)
                    self.fds[bi][pi][ci] = fd
        print("  Done.")

        # ── CPU Adam states ─────────────────────────────────────────────
        self.m      = []   # momentum (fp32, CPU)
        self.v      = []   # variance (fp32, CPU)
        self.p_fp32 = []   # fp32 master weights (CPU)
        self.grads  = []   # fp32 gradient accumulators (CPU)

        print("  Allocating CPU Adam states ...")
        for blk in blocks:
            ps = list(blk.parameters())
            p32s = [p.data.float().cpu() for p in ps]
            self.p_fp32.append(p32s)
            self.m.append([torch.zeros_like(x) for x in p32s])
            self.v.append([torch.zeros_like(x) for x in p32s])
            self.grads.append([torch.zeros_like(x) for x in p32s])
        print("  Done.")

    # ── Chunked write/read helpers ───────────────────────────────────────

    def _write_param(self, bi, pi, tensor):
        """Write one tensor to its chunk files on PM."""
        raw     = tensor.contiguous()
        n_bytes = raw.nbytes
        base    = raw.data_ptr()
        n_chunks = len(self.fds[bi][pi])
        for ci in range(n_chunks):
            byte_off    = ci * CHUNK_BYTES
            byte_end    = min(byte_off + CHUNK_BYTES, n_bytes)
            chunk_bytes = byte_end - byte_off
            chunk_ptr   = base + byte_off
            self._ext.write(self.fds[bi][pi][ci], chunk_ptr, chunk_bytes)

    def _read_param(self, bi, pi, tensor):
        """Read one tensor from its chunk files (DRAM hit after prefetch)."""
        raw     = tensor.contiguous()
        n_bytes = raw.nbytes
        base    = raw.data_ptr()
        n_chunks = len(self.fds[bi][pi])
        for ci in range(n_chunks):
            byte_off    = ci * CHUNK_BYTES
            byte_end    = min(byte_off + CHUNK_BYTES, n_bytes)
            chunk_bytes = byte_end - byte_off
            chunk_ptr   = base + byte_off
            self._ext.read(self.fds[bi][pi][ci], chunk_ptr, chunk_bytes)

    # ── Core interface (per e2e_experiment_plan.md) ──────────────────────

    def evict_readonly(self, block_idx, params):  # noqa: ARG002
        """
        Forward/Backward evict: params not modified, only release GPU memory.
        PM already has the correct version, no write needed.

        Args:
            block_idx: Block index (unused, kept for interface consistency)
            params: List of parameters (unused, kept for interface consistency)

        Note: In PyTorch, we can't truly "release" GPU memory without deleting
        the tensor. The caller should del params[i] after this call.
        This function is a no-op for Beaver (PM already has correct data).
        """
        _ = block_idx, params  # Silence unused parameter warnings

    def evict_dirty(self, block_idx, params):
        """
        Optimizer step evict: params modified, must COW write to PM.
        Also triggers async prefetch for next iteration.
        """
        t0 = time.perf_counter()
        for pi, p in enumerate(params):
            self._write_param(block_idx, pi, p.data)
        torch.cuda.synchronize()
        self._write_ms.append((time.perf_counter() - t0) * 1e3)

        # Trigger async prefetch (PM → DRAM) for next iteration
        self.prefetch(block_idx)

    def restore(self, block_idx, params):
        """Restore: DRAM hit → GPU (fast path after prefetch)."""
        t0 = time.perf_counter()
        for pi, p in enumerate(params):
            self._read_param(block_idx, pi, p.data)
        torch.cuda.synchronize()
        self._read_ms.append((time.perf_counter() - t0) * 1e3)

    def prefetch(self, block_idx):
        """Gap-period prefetch: PM → pinned CPU DRAM (cpu memcpy)."""
        for pi in range(len(self.fds[block_idx])):
            for fd in self.fds[block_idx][pi]:
                self._ext.prefetch(fd)

    def reset_dram_pool(self):
        """Reclaim DRAM pool pages. Call at start of each iteration's prefetch."""
        self._ext.reset_dram_pool()

    # ── Legacy interface (for backward compatibility) ────────────────────

    def write(self, block_idx, params):
        """Alias for evict_dirty (legacy interface)."""
        self.evict_dirty(block_idx, params)

    def read(self, block_idx, params):
        """Alias for restore (legacy interface)."""
        self.restore(block_idx, params)

    # ── Gradient collection + Adam ───────────────────────────────────────

    def collect_grad(self, block_idx, params):
        """Move GPU gradients to fp32 CPU buffers, free GPU grad memory."""
        for p, gbuf in zip(params, self.grads[block_idx]):
            if p.grad is not None:
                gbuf.copy_(p.grad.float().cpu())
                p.grad = None

    def begin_optimizer_step(self):
        """Increment step counter once per training iteration before per-layer adam_step_gpu."""
        self.step_count += 1

    def adam_step_gpu(self, block_idx, params):
        """
        GPU AdamW for one block.

        Optimizer states (fp32 master, m, v) live on CPU DRAM between steps.
        For each param, they are temporarily moved to GPU, Adam runs on GPU,
        then returned to CPU. Updated fp16 params remain on GPU so that the
        subsequent evict_dirty call writes a GPU-computed result to PM.

        This mirrors the per-shard GPU optimizer behavior in FSDP/ZeRO-3 (no
        CPU optimizer offload), adapted for single-GPU by loading one block's
        states at a time due to VRAM constraints.
        """
        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count

        for pi, p in enumerate(params):
            g_cpu = self.grads[block_idx][pi]
            if not g_cpu.any():
                continue

            device = p.device
            g   = g_cpu.to(device, non_blocking=True)
            p32 = self.p_fp32[block_idx][pi].to(device, non_blocking=True)
            m   = self.m[block_idx][pi].to(device, non_blocking=True)
            v   = self.v[block_idx][pi].to(device, non_blocking=True)

            # AdamW on GPU
            p32.mul_(1.0 - self.lr * self.wd)
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            p32.addcdiv_(m / bc1, (v / bc2).sqrt().add_(self.eps), value=-self.lr)

            # fp16 result stays on GPU; evict_dirty will write this to PM
            p.data.copy_(p32.half())

            # Return optimizer states to CPU DRAM
            self.p_fp32[block_idx][pi].copy_(p32.cpu())
            self.m[block_idx][pi].copy_(m.cpu())
            self.v[block_idx][pi].copy_(v.cpu())
            del p32, m, v, g

    def zero_grads(self):
        for i in range(self.num_blocks):
            for g in self.grads[i]:
                g.zero_()

    def cleanup(self):
        """Release all PM/DRAM resources."""
        self._ext.cleanup()

    # ── Stats ────────────────────────────────────────────────────────────

    def write_ms_stats(self):
        a = self._write_ms
        return (float(np.median(a)), float(np.percentile(a, 95))) if a else (0.0, 0.0)

    def read_ms_stats(self):
        a = self._read_ms
        return (float(np.median(a)), float(np.percentile(a, 95))) if a else (0.0, 0.0)


class BaselineManager:
    """
    Baseline: DeepSpeed ZeRO-3 style DRAM two-hop + per-checkpoint rename.

    Simulates the standard approach: parameters are written through DRAM to
    PM (ext4-dax on /mnt/pmem), with crash consistency guaranteed by
    per-checkpoint directory rename (all blocks written → fsync → rename).

    Write path (evict_dirty):
      GPU → cudaMemcpy D2H → pinned DRAM buffer
      + pwrite to tmp directory (no fsync/rename yet)

    Checkpoint commit (checkpoint_commit, called once after all evict_dirty):
      fsync each file in tmp dir → fsync tmp dir
      → rename tmp dir → latest dir → fsync parent

    Read path (restore):
      pinned DRAM buffer → GPU (normal operation, DRAM always has latest)
    """

    def __init__(self, blocks, pm_dir="/mnt/pmem/baseline",
                 lr=LR, weight_decay=WEIGHT_DECAY,
                 beta1=BETA1, beta2=BETA2, eps=EPS):
        self.num_blocks  = len(blocks)
        self.lr          = lr
        self.wd          = weight_decay
        self.beta1       = beta1
        self.beta2       = beta2
        self.eps         = eps
        self.step_count  = 0
        self.pm_dir      = pm_dir
        self._ckpt_gen   = 0       # alternating checkpoint generation (0/1)

        self._write_ms      = []   # per evict_dirty call (GPU→DRAM + pwrite)
        self._read_ms       = []   # per restore call (DRAM→GPU)
        self._checkpoint_ms = []   # per checkpoint_commit call

        # Create PM directory structure
        os.makedirs(pm_dir, exist_ok=True)

        # ── Compute sizing ──────────────────────────────────────────────
        self.shapes = []
        total_bytes = 0

        for blk in blocks:
            blk_shapes = []
            for p in blk.parameters():
                blk_shapes.append((p.data.shape, p.data.dtype))
                total_bytes += p.data.nbytes
            self.shapes.append(blk_shapes)

        print(f"BaselineManager sizing:")
        print(f"  blocks      : {self.num_blocks}")
        print(f"  total bytes : {total_bytes / 1024**3:.2f} GB")
        print(f"  PM dir      : {pm_dir}")

        # ── DRAM buffers (pinned for fast H2D/D2H) ──────────────────────
        self.dram_buffers = []  # dram_buffers[bi][pi] = pinned CPU tensor
        print("  Allocating pinned DRAM buffers ...")
        for blk in blocks:
            blk_bufs = []
            for p in blk.parameters():
                buf = torch.empty_like(p.data, device='cpu', pin_memory=True)
                blk_bufs.append(buf)
            self.dram_buffers.append(blk_bufs)
        print("  Done.")

        # ── CPU Adam states ─────────────────────────────────────────────
        self.m      = []
        self.v      = []
        self.p_fp32 = []
        self.grads  = []

        print("  Allocating CPU Adam states ...")
        for blk in blocks:
            ps = list(blk.parameters())
            p32s = [p.data.float().cpu() for p in ps]
            self.p_fp32.append(p32s)
            self.m.append([torch.zeros_like(x) for x in p32s])
            self.v.append([torch.zeros_like(x) for x in p32s])
            self.grads.append([torch.zeros_like(x) for x in p32s])
        print("  Done.")

    # ── Path helpers ─────────────────────────────────────────────────────

    def _tmp_dir(self):
        """Staging directory for current checkpoint writes."""
        return os.path.join(self.pm_dir, "ckpt_tmp")

    def _latest_dir(self):
        """Committed checkpoint directory."""
        return os.path.join(self.pm_dir, f"ckpt_{self._ckpt_gen}")

    def _prev_dir(self):
        """Previous generation checkpoint (to be replaced)."""
        return os.path.join(self.pm_dir, f"ckpt_{1 - self._ckpt_gen}")

    @staticmethod
    def _param_filename(bi, pi):
        return f"block_{bi}_param_{pi}.bin"

    # ── Core interface ───────────────────────────────────────────────────

    def evict_readonly(self, block_idx, params):
        """
        Forward/Backward evict: params not modified, DRAM buffer already has
        the correct version from the last evict_dirty. No write needed.
        """
        _ = block_idx, params

    def evict_dirty(self, block_idx, params):
        """
        Optimizer step evict: params modified.
        GPU → pinned DRAM (cudaMemcpy D2H) + pwrite to tmp directory.
        Does NOT fsync or rename — that happens in checkpoint_commit().
        """
        t0 = time.perf_counter()

        tmp_dir = self._tmp_dir()
        os.makedirs(tmp_dir, exist_ok=True)

        for pi, p in enumerate(params):
            # GPU → pinned DRAM
            self.dram_buffers[block_idx][pi].copy_(p.data)

            # DRAM → PM tmp file (pwrite, no fsync yet)
            fpath = os.path.join(tmp_dir, self._param_filename(block_idx, pi))
            buf = self.dram_buffers[block_idx][pi]
            with open(fpath, 'wb') as f:
                f.write(buf.numpy().tobytes())

        torch.cuda.synchronize()
        self._write_ms.append((time.perf_counter() - t0) * 1e3)

    def checkpoint_commit(self):
        """
        Per-checkpoint crash consistency: fsync all files → rename directory.

        Called once after all blocks have been evict_dirty'd.
        Simulates DeepSpeed checkpoint: the entire checkpoint is atomic.

        Uses alternating generations (ckpt_0 / ckpt_1) so the previous
        committed checkpoint survives until the new one is fully committed.
        """
        t0 = time.perf_counter()

        tmp_dir = self._tmp_dir()
        new_dir = self._latest_dir()

        # 1. fsync every file in tmp directory
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            fd = os.open(fpath, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)

        # 2. fsync the tmp directory itself (directory entries)
        dir_fd = os.open(tmp_dir, os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)

        # 3. Remove old generation if it exists (to free the name for rename)
        import shutil
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)

        # 4. Atomic rename: tmp → new generation
        os.rename(tmp_dir, new_dir)

        # 5. fsync parent directory to persist the rename
        parent_fd = os.open(self.pm_dir, os.O_RDONLY)
        os.fsync(parent_fd)
        os.close(parent_fd)

        # Alternate generation for next checkpoint
        self._ckpt_gen = 1 - self._ckpt_gen

        self._checkpoint_ms.append((time.perf_counter() - t0) * 1e3)

    def restore(self, block_idx, params):
        """
        Restore: pinned DRAM buffer → GPU.
        Normal operation: DRAM always has latest data (updated by evict_dirty).
        """
        t0 = time.perf_counter()
        for pi, p in enumerate(params):
            p.data.copy_(self.dram_buffers[block_idx][pi])
        torch.cuda.synchronize()
        self._read_ms.append((time.perf_counter() - t0) * 1e3)

    def prefetch(self, block_idx):
        """No-op: DRAM buffers already have latest data."""
        pass

    def reset_dram_pool(self):
        """No-op for baseline (DRAM buffers are persistent)."""
        pass

    # ── Gradient collection + Adam ───────────────────────────────────────

    def collect_grad(self, block_idx, params):
        for p, gbuf in zip(params, self.grads[block_idx]):
            if p.grad is not None:
                gbuf.copy_(p.grad.float().cpu())
                p.grad = None

    def begin_optimizer_step(self):
        """Increment step counter once per training iteration before per-layer adam_step_gpu."""
        self.step_count += 1

    def adam_step_gpu(self, block_idx, params):
        """GPU AdamW for one block. See BeaverManager.adam_step_gpu for full docstring."""
        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count

        for pi, p in enumerate(params):
            g_cpu = self.grads[block_idx][pi]
            if not g_cpu.any():
                continue

            device = p.device
            g   = g_cpu.to(device, non_blocking=True)
            p32 = self.p_fp32[block_idx][pi].to(device, non_blocking=True)
            m   = self.m[block_idx][pi].to(device, non_blocking=True)
            v   = self.v[block_idx][pi].to(device, non_blocking=True)

            p32.mul_(1.0 - self.lr * self.wd)
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            p32.addcdiv_(m / bc1, (v / bc2).sqrt().add_(self.eps), value=-self.lr)

            p.data.copy_(p32.half())

            self.p_fp32[block_idx][pi].copy_(p32.cpu())
            self.m[block_idx][pi].copy_(m.cpu())
            self.v[block_idx][pi].copy_(v.cpu())
            del p32, m, v, g

    def zero_grads(self):
        for i in range(self.num_blocks):
            for g in self.grads[i]:
                g.zero_()

    def cleanup(self):
        """Clean up PM files."""
        import shutil
        if os.path.exists(self.pm_dir):
            shutil.rmtree(self.pm_dir)

    # ── Stats ────────────────────────────────────────────────────────────

    def write_ms_stats(self):
        a = self._write_ms
        return (float(np.median(a)), float(np.percentile(a, 95))) if a else (0.0, 0.0)

    def read_ms_stats(self):
        a = self._read_ms
        return (float(np.median(a)), float(np.percentile(a, 95))) if a else (0.0, 0.0)

    def checkpoint_ms_stats(self):
        a = self._checkpoint_ms
        return (float(np.median(a)), float(np.percentile(a, 95))) if a else (0.0, 0.0)


# Backward compatibility alias
BeaverOffloadManager = BeaverManager

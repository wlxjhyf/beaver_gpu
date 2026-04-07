"""
gpufs_offload.py — GPUfsManager for E2e GPUfs comparison experiment.

Write path (evict_dirty):
  GPU tensor → gpufs_ext.write() → GPU kernel calls gwrite() →
  CPU handler pwrite() → /mnt/pmem/gpufs/ckpt_tmp/<file>

Read path (restore):
  gpufs_ext.read() → GPU kernel calls gread() →
  CPU handler cudaMemcpy H2D → GPU tensor

Crash consistency: same fsync+rename per-checkpoint as BaselineManager.

Interface is identical to BaselineManager so the training script can
swap managers with a one-line change.
"""

import os
import shutil
import time
import numpy as np
import torch


def _import_gpufs_ext():
    try:
        import gpufs_ext
        return gpufs_ext
    except ImportError as e:
        raise ImportError(
            "gpufs_ext not found. Build it first:\n"
            "  cd beaver_torch && python setup_gpufs.py build_ext --inplace\n"
            f"Original error: {e}"
        ) from e


# Adam hyperparams (must match training script)
LR           = 3e-4
WEIGHT_DECAY = 0.01
BETA1, BETA2 = 0.9, 0.999
EPS          = 1e-8


class GPUfsManager:
    """
    Per-layer parameter offload to PM via GPUfs (GPU-side RPC I/O).

    Write path (evict_dirty):
      GPU tensor → gpufs_ext.write() → GPU IPC → CPU pwrite() → /mnt/pmem

    Read path (restore):
      gpufs_ext.read() → CPU pread() → GPU staging → GPU tensor

    Crash consistency: same fsync+rename per-checkpoint as BaselineManager.
    No pinned DRAM buffer needed (GPUfs manages its own page pool in VRAM).
    """

    def __init__(self, blocks, pm_dir="/mnt/pmem/gpufs",
                 lr=LR, weight_decay=WEIGHT_DECAY,
                 beta1=BETA1, beta2=BETA2, eps=EPS):
        self._ext        = _import_gpufs_ext()
        self.num_blocks  = len(blocks)
        self.lr          = lr
        self.wd          = weight_decay
        self.beta1       = beta1
        self.beta2       = beta2
        self.eps         = eps
        self.step_count  = 0
        self.pm_dir      = pm_dir
        self._ckpt_gen   = 0

        self._write_ms      = []
        self._read_ms       = []
        self._checkpoint_ms = []

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

        print(f"GPUfsManager sizing:")
        print(f"  blocks      : {self.num_blocks}")
        print(f"  total bytes : {total_bytes / 1024**3:.2f} GB")
        print(f"  PM dir      : {pm_dir}")

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
        return os.path.join(self.pm_dir, "ckpt_tmp")

    def _latest_dir(self):
        return os.path.join(self.pm_dir, f"ckpt_{self._ckpt_gen}")

    @staticmethod
    def _param_filename(bi, pi):
        return f"block_{bi}_param_{pi}.bin"

    # ── Core interface ───────────────────────────────────────────────────

    def evict_readonly(self, block_idx, params):
        """Forward/Backward evict: params not modified, no write needed."""
        _ = block_idx, params

    def evict_dirty(self, block_idx, params):
        """
        Optimizer step evict: GPU tensor → PM via GPUfs gwrite RPC.
        Does NOT fsync/rename — that happens in checkpoint_commit().
        """
        t0 = time.perf_counter()

        tmp_dir = self._tmp_dir()
        os.makedirs(tmp_dir, exist_ok=True)

        for pi, p in enumerate(params):
            fpath = os.path.join(tmp_dir,
                                 self._param_filename(block_idx, pi))
            raw = p.data.contiguous()
            self._ext.write(fpath.encode(), raw.data_ptr(), raw.nbytes)

        torch.cuda.synchronize()
        self._write_ms.append((time.perf_counter() - t0) * 1e3)

    def checkpoint_commit(self):
        """
        Per-checkpoint crash consistency: fsync all files → rename directory.
        Identical to BaselineManager.checkpoint_commit.
        """
        t0 = time.perf_counter()

        tmp_dir = self._tmp_dir()
        new_dir = self._latest_dir()

        # 1. fsync every file
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            fd = os.open(fpath, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)

        # 2. fsync the tmp directory
        dir_fd = os.open(tmp_dir, os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)

        # 3. Remove old generation if it exists
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)

        # 4. Atomic rename
        os.rename(tmp_dir, new_dir)

        # 5. fsync parent
        parent_fd = os.open(self.pm_dir, os.O_RDONLY)
        os.fsync(parent_fd)
        os.close(parent_fd)

        self._ckpt_gen = 1 - self._ckpt_gen
        self._checkpoint_ms.append((time.perf_counter() - t0) * 1e3)

    def restore(self, block_idx, params):
        """Restore: PM → GPU via GPUfs gread RPC."""
        t0 = time.perf_counter()

        ckpt_dir = self._latest_dir()
        for pi, p in enumerate(params):
            fpath = os.path.join(ckpt_dir,
                                 self._param_filename(block_idx, pi))
            raw = p.data.contiguous()
            self._ext.read(fpath.encode(), raw.data_ptr(), raw.nbytes)

        torch.cuda.synchronize()
        self._read_ms.append((time.perf_counter() - t0) * 1e3)

    def prefetch(self, block_idx):
        """No-op: GPUfs does not support async PM→DRAM prefetch."""
        pass

    def reset_dram_pool(self):
        """No-op: GPUfs has no DRAM pool managed at this level."""
        pass

    # ── Gradient collection + Adam ───────────────────────────────────────

    def collect_grad(self, block_idx, params):
        for p, gbuf in zip(params, self.grads[block_idx]):
            if p.grad is not None:
                gbuf.copy_(p.grad.float().cpu())
                p.grad = None

    def begin_optimizer_step(self):
        self.step_count += 1

    def adam_step_gpu(self, block_idx, params):
        """GPU AdamW for one block. See BeaverManager.adam_step_gpu."""
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
        """Release GPUfs state and clean up PM files."""
        self._ext.cleanup()
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

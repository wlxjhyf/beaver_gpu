"""
beaver_offload.py — BeaverOffloadManager: PM-backed per-layer offload.

Drop-in replacement for OffloadManager in e2e_train_offload.py.
Uses Beaver GPU F2FS + Beaver COW for crash-consistent PM storage.

Write path:  GPU tensor → Beaver COW → PM  (via beaver_ext.write)
Gap:         PM → pinned DRAM              (via beaver_ext.prefetch)
Read path:   pinned DRAM → GPU tensor     (via beaver_ext.read, DRAM hit)

File chunking:
  F2FS_ADDRS_PER_INODE = 1017 → max 1017 pages (~4.16 MiB) per file.
  Large tensors are split into multiple chunk files automatically.
  Each chunk is written/read independently.

Usage in e2e_train_offload.py:
  from beaver_offload import BeaverOffloadManager
  offload = BeaverOffloadManager(blocks)
  # Same interface as OffloadManager: write/prefetch/read/adam_step/...
"""

import os
import math
import time
import numpy as np
import torch

# Adam hyperparams (must match e2e_train_offload.py)
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


class BeaverOffloadManager:
    """
    Per-layer parameter offload to Intel Optane PM via Beaver COW.

    Interface is identical to OffloadManager (DRAM version) in
    e2e_train_offload.py so the training script can switch backends
    by changing one line.

    Init computes FS sizing from actual tensor shapes and pre-creates
    all chunk files on the PM-backed GPU F2FS.
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

        print(f"BeaverOffloadManager sizing:")
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

    # ── Chunked write helper ─────────────────────────────────────────────

    def _write_param(self, bi, pi, tensor):
        """Write one tensor to its chunk files on PM."""
        raw     = tensor.contiguous()
        n_bytes = raw.nbytes
        base    = raw.data_ptr()          # CUDA device pointer to start of storage
        n_chunks = len(self.fds[bi][pi])
        for ci in range(n_chunks):
            byte_off    = ci * CHUNK_BYTES
            byte_end    = min(byte_off + CHUNK_BYTES, n_bytes)
            chunk_bytes = byte_end - byte_off
            # Compute pointer by offsetting base address directly.
            # raw.view(uint8)[byte_off:].data_ptr() returns 0 for non-zero offsets
            # on CUDA tensors (PyTorch view+slice behavior); use base+offset instead.
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

    # ── Storage interface (PM-swap point) ───────────────────────────────

    def write(self, block_idx, params):
        """Evict: GPU → PM via Beaver COW (crash-consistent)."""
        t0 = time.perf_counter()
        for pi, p in enumerate(params):
            self._write_param(block_idx, pi, p.data)
        torch.cuda.synchronize()
        self._write_ms.append((time.perf_counter() - t0) * 1e3)

    def prefetch(self, block_idx):
        """Gap-period prefetch: PM → pinned CPU DRAM (cpu memcpy)."""
        for pi in range(len(self.fds[block_idx])):
            for fd in self.fds[block_idx][pi]:
                self._ext.prefetch(fd)

    def read(self, block_idx, params):
        """Restore: DRAM hit → GPU (fast path after prefetch)."""
        t0 = time.perf_counter()
        for pi, p in enumerate(params):
            self._read_param(block_idx, pi, p.data)
        torch.cuda.synchronize()
        self._read_ms.append((time.perf_counter() - t0) * 1e3)

    def reset_dram_pool(self):
        """Reclaim DRAM pool pages. Call at start of each iteration's prefetch."""
        self._ext.reset_dram_pool()

    # ── Gradient collection + Adam ───────────────────────────────────────

    def collect_grad(self, block_idx, params):
        """Move GPU gradients to fp32 CPU buffers, free GPU grad memory."""
        for p, gbuf in zip(params, self.grads[block_idx]):
            if p.grad is not None:
                gbuf.copy_(p.grad.float().cpu())
                p.grad = None

    def adam_step(self):
        """CPU AdamW step across all blocks."""
        self.step_count += 1
        t   = self.step_count
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t

        for i in range(self.num_blocks):
            for p32, m, v, g in zip(
                    self.p_fp32[i], self.m[i], self.v[i], self.grads[i]):
                if not g.any():
                    continue
                p32.mul_(1.0 - self.lr * self.wd)
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                m_hat = m / bc1
                v_hat = v / bc2
                p32.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def sync_weights_to_gpu(self, blocks):
        """
        After adam_step, push updated fp32 master weights back to GPU fp16 params.
        Call this after adam_step() and before the next forward pass.
        Unlike the DRAM path (which syncs via pinned buffers), here the updated
        weights must be written to PM and then read back to GPU.

        Practical use:
          offload.adam_step()
          offload.sync_weights_to_gpu(blocks)   # update GPU params from fp32
          offload.reset_dram_pool()
          for i in range(num_blks):
              offload.prefetch(i)
        """
        for bi, blk in enumerate(blocks):
            for pi, p in enumerate(blk.parameters()):
                p32 = self.p_fp32[bi][pi]
                # Update GPU fp16 param from CPU fp32 master
                p.data.copy_(p32.half().to(p.device))

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

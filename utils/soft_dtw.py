# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Source: https://github.com/Maghoumi/pytorch-softdtw-cuda

import math

import numpy as np
import torch
from numba import jit, prange
from torch.autograd import Function

# Try to import CUDA support
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


if CUDA_AVAILABLE:
    @cuda.jit
    def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        I = tid
        inv_gamma = 1.0 / gamma

        for p in range(n_passes):
            J = max(0, min(p - tid, max_j - 1))
            i = I + 1
            j = J + 1

            if I + J == p and (I < max_i and J < max_j):
                if not (abs(i - j) > bandwidth > 0):
                    r0 = -R[b, i - 1, j - 1] * inv_gamma
                    r1 = -R[b, i - 1, j] * inv_gamma
                    r2 = -R[b, i, j - 1] * inv_gamma
                    rmax = max(max(r0, r1), r2)
                    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin

            cuda.syncthreads()

    @cuda.jit
    def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
        k = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        I = tid

        for p in range(n_passes):
            rev_p = n_passes - p - 1
            J = max(0, min(rev_p - tid, max_j - 1))
            i = I + 1
            j = J + 1

            if I + J == rev_p and (I < max_i and J < max_j):
                if math.isinf(R[k, i, j]):
                    R[k, i, j] = -math.inf

                if not (abs(i - j) > bandwidth > 0):
                    a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                    b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                    c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                    E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

            cuda.syncthreads()


    class _SoftDTWCUDA(Function):
        @staticmethod
        def forward(ctx, D, gamma, bandwidth):
            dev = D.device
            dtype = D.dtype
            gamma = torch.cuda.FloatTensor([gamma])
            bandwidth = torch.cuda.FloatTensor([bandwidth])

            B = D.shape[0]
            N = D.shape[1]
            M = D.shape[2]
            threads_per_block = max(N, M)
            n_passes = 2 * threads_per_block - 1

            R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
            R[:, 0, 0] = 0

            compute_softdtw_cuda[B, threads_per_block](
                cuda.as_cuda_array(D.detach()),
                gamma.item(), bandwidth.item(), N, M, n_passes,
                cuda.as_cuda_array(R)
            )
            ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
            return R[:, -2, -2]

        @staticmethod
        def backward(ctx, grad_output):
            dev = grad_output.device
            dtype = grad_output.dtype
            D, R, gamma, bandwidth = ctx.saved_tensors

            B = D.shape[0]
            N = D.shape[1]
            M = D.shape[2]
            threads_per_block = max(N, M)
            n_passes = 2 * threads_per_block - 1

            D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
            D_[:, 1:N + 1, 1:M + 1] = D

            R[:, :, -1] = -math.inf
            R[:, -1, :] = -math.inf
            R[:, -1, -1] = R[:, -2, -2]

            E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
            E[:, -1, -1] = 1

            compute_softdtw_backward_cuda[B, threads_per_block](
                cuda.as_cuda_array(D_),
                cuda.as_cuda_array(R),
                1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                cuda.as_cuda_array(E)
            )
            E = E[:, 1:N + 1, 1:M + 1]
            return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# CPU implementation
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < np.abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                if 0 < bandwidth < np.abs(i - j):
                    continue
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


class SoftDTW(torch.nn.Module):
    """
    Soft-DTW loss for time series alignment.

    Supports both CPU and CUDA implementations with automatic fallback.
    """

    def __init__(self, use_cuda=True, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Args:
            use_cuda: Use CUDA if available (auto-fallback to CPU if not)
            gamma: Smoothing parameter (lower = closer to hard DTW)
            normalize: Normalize the distance (makes it a proper metric)
            bandwidth: Sakoe-Chiba bandwidth for pruning (None = no pruning)
            dist_func: Custom distance function (default: squared Euclidean)
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.dist_func = dist_func if dist_func is not None else SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by
        assert dx == dy

        use_cuda = self.use_cuda and x.is_cuda

        if use_cuda and (lx > 1024 or ly > 1024):
            use_cuda = False

        if use_cuda and CUDA_AVAILABLE:
            return _SoftDTWCUDA.apply
        return _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute soft-DTW between X and Y.

        Args:
            X: (batch_size, seq_len, dims)
            Y: (batch_size, seq_len, dims)

        Returns:
            Soft-DTW distances (batch_size,)
        """
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)


def soft_dtw_loss(pred, target, gamma=0.1, normalize=True):
    """
    Convenience function for computing soft-DTW loss on 1D signals.

    Args:
        pred: Predicted signal (B, T)
        target: Target signal (B, T)
        gamma: Smoothing parameter (lower = closer to hard DTW)
        normalize: Whether to normalize (recommended for loss)

    Returns:
        Mean soft-DTW loss (scalar), normalized by sequence length to be
        comparable in magnitude to MSE loss.
    """
    # Add feature dimension: (B, T) -> (B, T, 1)
    pred_3d = pred.unsqueeze(-1)
    target_3d = target.unsqueeze(-1)

    use_cuda = pred.is_cuda and CUDA_AVAILABLE
    sdtw = SoftDTW(use_cuda=use_cuda, gamma=gamma, normalize=normalize)

    # Normalize by sequence length to make loss comparable to MSE
    # Raw soft-DTW scales with sequence length, so divide by T
    seq_len = pred.shape[-1]
    return sdtw(pred_3d, target_3d).mean() / seq_len

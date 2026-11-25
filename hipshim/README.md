# hipshim: build SBD's Thrust backend for AMD GPUs with `hipcc`

SBD's Thrust backend is written against CUDA, Thrust, CUB, cuBLAS and NCCL. ROCm provides compatible libraries
(rocThrust, hipCUB, hipBLAS, RCCL) under different names. These headers map the names, so the backend compiles with
`hipcc` without editing its sources. They are put ahead of the system headers with `-I hipshim` and
`-include hipshim/thrust_cuda.h` (see `apps/chemistry_tpb_selected_basis_diagonalization/Makefile.hip`).

| header | maps |
|---|---|
| `cuda_runtime.h`, `cuda.h` | the CUDA runtime names SBD uses (`cudaMalloc`, streams, events, `cudaFuncSetAttribute`, ...) to HIP; sets `CUDART_VERSION 12000` so the pre-CUDA-13 `cudaMemAdvise(ptr, n, advice, device)` branch is used |
| `cublas_v2.h` | hipBLAS: the cuBLAS calls of `SBD_USE_CUBLAS` (argument order is the same) |
| `nccl.h` | RCCL, which keeps NCCL's API (`ncclComm_t`, `ncclBroadcast`, `ncclSend`/`ncclRecv`, groups) |
| `cub/warp/warp_reduce.cuh` | hipCUB (`namespace cub = hipcub`) |
| `thrust_cuda.h` (force-included) | `thrust::cuda` → `thrust::hip`; marks rocThrust's OpenMP backend usable (HIP's device pass does not define `_OPENMP`, and SBD's host-only `thrust::omp` calls would otherwise trip a static assertion) |

Device code that has no HIP equivalent (inline PTX) is handled in the sources, behind `__HIP_PLATFORM_AMD__`.

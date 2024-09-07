# Triton
Building Triton and CUDA side-by-side in order to create a cuBLAS-performant GEMM kernel.

## Setting Up Your Triton Development Enviornment
1. **cd into `triton_vs_cuda/triton`**
2. **Simply launch the included Docker image**
```
docker run -it --rm --gpus=all  --volume "$PWD/:/home/workspace" --workdir /home/workspace triton
docker build -t triton .
```

3. Once in your working directory, ensure that triton works by running `python3 tests/sgemm.py`

You should end up seeing something like this outputted in your terminal:
```
Matrix size: 2056x2056 and 2056x2056
PyTorch SGEMM time: 0.1012 seconds
Triton SGEMM time: 2.2521 seconds
```
# Sources
- https://triton-lang.org/main/getting-started/installation.html
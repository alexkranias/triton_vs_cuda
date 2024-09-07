# triton_vs_cuda
**Building Triton and CUDA kernels side-by-side in order to create a cuBLAS-performant GEMM kernel.**

Lately I've been learning Triton, its strengths, and its weaknesses. Inspired by [SiBohem's blog](https://siboehm.com/articles/22/CUDA-MMM), I thought I would show how we can attempt to build a Triton kernel as performant as a near-cuBLAS performant CUDA kernel. In this endeavor I hope to highlight a few things about Triton:
- what are the limitations of a Triton's block level programming paradigm?
- as a kernel engineer, how much control do we retain in Triton to squeeze more performance out?
- where does the Triton compiler take over and attempt to fill in? How successful is it at this task? Where is work still needed at the compiler level?
- when should you _actually_ use Triton v.s. CUDA?

## Getting Started
I've divided this project into two branches:
- `main`: template kernel files
- `solutions`: solution kernel files

I've included dockerfiles in each `/triton` and `/cuda` directory to make enviornment setup quick and easy. Open those directories and you'll find `README.md`s explaining how to get going.

### In Progress
I'll have a blog on the subject posted at some point on my personal website: [**alexkranias.com**](https://alexkranias.com)

_I'm actively working on that piece._

In the meantime, you can `clone` this repo to work on this on your own and follow SiBohem's blog.


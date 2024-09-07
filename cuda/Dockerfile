# IMPORTANT: use nvidia-smi to check your machine's CUDA version.
# If your machine's CUDA version is older than the image, the image
# will not work. Your machine's version must be equal to or newer.

# Below we use CUDA 12.2.0

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /home/workspace

RUN apt update
RUN apt install -y cmake
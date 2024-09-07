# CUDA
Building Triton and CUDA side-by-side in order to create a cuBLAS-performant GEMM kernel.

# Setting up your CUDA development enviornment

**Windows Users:** I highly recommend getting the most updated version of your NVIDIA drivers, Docker Desktop, and WSL. You may run into problems that can be fixed by updating to the latest version of one if not all of these three things.

## 1. We will be using Docker. Please install the Docker Engine / Docker Desktop for your OS 
Install: https://docs.docker.com/engine/install/

## 2. Verify you have NVIDIA Drivers installed. Type and enter `nvidia-smi` in your terminal and check that stuff shows up.
```
Sun Aug 25 17:11:57 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.81                 Driver Version: 560.81         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
```
If it doesn't exist you'll need to install NVIDIA Drivers: https://www.nvidia.com/en-us/drivers/

## 3. Install the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) which is required to run CUDA Docker images.

**Windows Users:** you will need to update to WSL2. In a new Powershell/Terminal window execute the following:
```
wsl --update
```
Once complete check if your WSL version is 2.0 or newer.
```
wsl --version
```
You should see something like `WSL version: 2.2.4.0`

Find more information: https://docs.docker.com/desktop/gpu/

## 4. Launch a Docker container using the included Dockerfile in `/cuda`
1. cd into `triton_vs_cuda/cuda`
2. run
```
docker build -t cuda .
docker run -it --rm --gpus=all  --volume "$PWD/:/home/workspace" --workdir /home/workspace triton
```
## 5. Build the project
```
mkdir build
cd build
cmake ..
cmake --build .
```

If you run into a CMake issue. Once you add your fix, I recommend deleting your build, then rebuilding with `cmake ..`


# Manually launch CUDA docker container
## 1. Find the [latest CUDA Development Docker Image](https://hub.docker.com/r/nvidia/cuda) and pull it.
```ssh
docker pull nvidia/cuda:12.6.0-devel-ubuntu24.04
```
**IMPORTANT NOTE:** the CUDA image you pull must have a version **equal to or LESS than** the CUDA Driver version installed on your machine. Check what version you have using `nvcc --version`.

## 2. Open the CUDA directory to initialize your workspace in: `cd cuda`

## 3. Launch the Docker image. It will launch a workspace in your PWD.
```
docker run -it --rm --gpus=all  --volume "$PWD/:/home/workspace" --workdir /home/workspace  nvidia/cuda:12.6.0-devel-ubuntu24.04
```

You should now be in your Docker image (e.g. `root@7efed242491b:\home\workspace#`)

## 4. Confirm GPU is accessible within Docker container
```
nvidia-smi
```

You should see your GPU.
```
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   50C    P8              9W /   65W |     555MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
If you don't this means that your **NVIDIA Container Toolkit (or WSL2) is not working/installed.**

## 5. Install Other Libraries You'll Need in Your Docker Container
```
apt update
apt install cmake
```

## 6. Setup `build` dir and build
```
mkdir build
cd build
cmake ..
cmake --build .
```

If you run into a CMake issue. Once you add your fix, I recommend deleting your build, then rebuilding with `cmake ..`

# Useful Resources
- Tools/Misc
    - https://excalidraw.com/
- NVIDIA/CUDA Setup
    - https://medium.com/@u.mele.coding/a-beginners-guide-to-nvidia-container-toolkit-on-docker-92b645f92006
    - https://hub.docker.com/r/nvidia/cuda


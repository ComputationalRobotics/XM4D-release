#!/usr/bin/env bash
# Conda env + build for step 2 of a photo collection (Python/match_photos.py): our GLOMAP fork (glomap/) and the COLMAP
# 3.11 it embeds (CMake FetchContent, commit 32542632). The fork writes the view graph (tracks) to ./assets/tempdata/
# (relative to the working directory) instead of running global positioning, and makes rotation averaging robust to
# failed IRLS/L1 steps; it does not compile against newer COLMAP releases.
# Builds, in build/glomap/:  glomap/glomap  and  _deps/colmap-build/src/colmap/exe/colmap  (CUDA SIFT, no GUI).
# Everything the build needs comes from conda-forge into the env: gcc 12 (the newest that nvcc 12.4 accepts), CMake,
# CUDA 12.4 (nvcc, cudart, curand), Ceres, Boost, FreeImage, ... No CUDA toolkit or sudo is needed, only an NVIDIA
# driver for CUDA >= 12.4 (>= 550). The binaries find their libraries in the env through their RPATH, so they run
# without activating the env, but the env must stay installed.
# CUDA archs: machine code for Volta to Hopper (sm_70-90: V100, RTX 20xx-40xx, A100, L40, H100) plus PTX for newer GPUs,
# so the build does not depend on the GPU of the build machine.
# Needs network (conda-forge, github; ~0.4 GB of downloads). Takes ~2 min with 128 cores (the compile is ~25 CPU-min, so
# ~10 min with 4 cores). Disk: env 1.8 GB, build 0.4 GB, plus 1.4 GB in conda's package cache (conda clean -a frees it).
# usage: bash install/glomap.sh [env name]      (default env name: glomap)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-glomap}"

# Ceres 2.1: the conda-forge Ceres 2.2 config looks for SuiteSparse 7 and clashes with glomap/cmake/FindSuiteSparse
new_env "$ENV" -c conda-forge --override-channels \
    cmake=3.31 ninja gcc_linux-64=12 gxx_linux-64=12 cuda-version=12.4 cuda-nvcc cuda-cudart-dev libcurand-dev \
    eigen=3.4.0 ceres-solver=2.1.0 suitesparse=5.10 glog=0.6 gflags=2.2 libboost-devel=1.86 freeimage=3.18 \
    flann=1.9 metis=5.1 lz4-c sqlite glew libgl-devel libopengl-devel

B="$ROOT/build/glomap"
rm -rf "$B"
# libraries from the env first; nvcc compiles the archs in parallel (--threads 0); MARCH_NATIVE=OFF builds PoseLib
# without -march=native, whose -Werror fails with gcc 12 on AVX-512 CPUs (and keeps the binaries portable)
cmake -S "$ROOT/glomap" -B "$B" -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_FLAGS="--threads 0" \
    -DCMAKE_CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;89-real;90" \
    -DFETCH_COLMAP=ON -DFETCH_POSELIB=ON -DMARCH_NATIVE=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DTESTS_ENABLED=OFF \
    -DCCACHE_ENABLED=OFF
cmake --build "$B" --target glomap_main colmap_main

COLMAP="$B/_deps/colmap-build/src/colmap/exe/colmap"
# COLMAP's CMake silently drops CUDA when it does not find it
grep -q "with CUDA" <<< "$("$COLMAP" help)" || { echo "[install] COLMAP was built without CUDA" >&2; exit 1; }
echo "[install] glomap: $B/glomap/glomap"
echo "[install] colmap: $COLMAP"

installed glomap

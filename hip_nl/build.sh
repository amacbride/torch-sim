#!/bin/bash
# Build script for hip_nl library

set -e  # Exit on error

echo "Building hip_nl library..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Copy library to parent directory
cp libhip_nl.so ..

cd ..
echo "Build complete! Library: libhip_nl.so"
echo ""
echo "Test the installation with:"
echo "  python -c 'import hip_nl; print(hip_nl.HIP_NL_AVAILABLE)'"

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ implementation of shape-based matching (also known as LINE-MOD), a computer vision algorithm for detecting textureless objects in images using gradient orientation features. The implementation improves upon OpenCV's LINE-MOD with better performance and additional features.

## Core Architecture

- **line2Dup.h/cpp**: Main implementation of the shape-based matching algorithm
  - `line2Dup::Detector`: Primary class for template matching operations
  - `ColorGradientPyramid`: Handles multi-scale gradient computation and feature extraction
  - `Template`: Stores extracted features for matching
  - `Match`: Represents detection results with position, similarity score, and template ID

- **shape_based_matching namespace**: Utilities for template generation
  - `shapeInfo_producer`: Generates training templates with rotation and scaling variations
  - Handles angle/scale ranges and produces transformation matrices

- **MIPP library**: Cross-platform SIMD optimization library for x86 SSE/AVX and ARM NEON

## Build Commands

```bash
# Build the project
mkdir build && cd build
cmake ..
make

# Run tests
./shape_based_matching_test
```

## Development Setup

1. **Prerequisites**: 
   - OpenCV 3.x required
   - CMake 2.8+
   - C++14 compiler

2. **Configuration**:
   - Update `test.cpp:9` prefix path to your working directory
   - Modify `CMakeLists.txt:29` OpenCV path if needed (default: `/opt/ros/kinetic`)

3. **SIMD Support**:
   - Automatically detects platform (ARM NEON vs x86 SSE/AVX)
   - Can disable SIMD with `-DMIPP_NO_INTRINSICS` flag

## Key Algorithms

- **Multi-pyramid matching**: Uses 2 pyramid levels with strides 4 and 8 for speed
- **Gradient orientation**: Extracts 8-direction quantized gradients (0-7 labels)
- **NMS (Non-Maximum Suppression)**: Built-in implementation for filtering overlapping detections
- **Template rotation**: Direct feature rotation for efficient template generation

## Test Cases Structure

- **case0/**: Circle detection with scale variations
- **case1/**: Arbitrary shape with rotation testing  
- **case2/**: Noise robustness testing

Template files (`.yaml`) store extracted features, info files store transformation parameters.

## Performance Notes

- Input images should have dimensions as multiples of 32 for optimal performance
- Typical performance: 1000 templates processed in ~20ms
- Feature extraction uses SIMD for 4x+ speedup on supported platforms
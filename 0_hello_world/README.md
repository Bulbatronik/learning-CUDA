# Learning CUDA

A simple "Hello World" program in CUDA C++.

## Prerequisites

- NVIDIA CUDA Toolkit installed
- NVIDIA GPU with CUDA support (for GPU execution)

## How to Run

1. **Compile the program:**
   ```bash
   nvcc -o main main.cu
   ```

2. **Run the executable:**
   ```bash
   ./main
   ```

## Expected Output

```
hello world
```

## Files

- `main.cu` - The main CUDA C++ source file containing a simple hello world program

## Notes

- This is a basic C++ program that doesn't use GPU features yet
- The `.cu` extension indicates it's a CUDA source file
- `nvcc` is the NVIDIA CUDA compiler

# Work-Stealing Thread Pool

This repository implements a high-performance, work-stealing thread pool in modern C++20
and demonstrates its use with a parallel 3D volumetric convolution example (Gaussian blur,
Laplacian, and directional edge detection).

## Quick Start

Compile and run the demo:

```bash
g++ -std=c++20 -O3 -pthread src/3d_convolution/main.cpp -o demo
./demo
```

Generate the Doxygen HTML documentation and open the index:

```bash
doxygen Doxyfile
xdg-open docs/doxygen/html/index.html
```

## Key Features

- Work-stealing thread pool using `std::jthread` and `std::stop_token`
- Thread-safe deque primitive (`ThreadSafeDeque`) supporting owner LIFO and stealer FIFO
- Parallel 3D convolution with task decomposition per depth slice
- Clear examples of modern C++ concurrency and RAII patterns

## Project Layout

- `src/core/thread_pool.hpp` — work-stealing thread pool implementation
- `src/core/thread_safe_deque.hpp` — thread-safe deque implementation
- `src/3d_convolution/convolution.hpp` — convolution task and helpers
- `src/3d_convolution/main.cpp` — demo entry point
- `Doxyfile` — Doxygen configuration

## 3D Convolution Use Case

The demo synthesizes a 24×24×24 volumetric image with a central high-intensity cube and
Gaussian noise, then applies 3×3×3 filters in parallel (one task per z-slice). This
demonstrates how to decompose a volumetric computation into parallel tasks and how
work-stealing balances load across worker threads.

## Notes

- The demo caps worker threads to a small number for predictable output; adjust in
	`ThreadPool` constructor if you want to benchmark on larger core counts.
- The `README.md` is used as the Doxygen main page (`USE_MDFILE_AS_MAINPAGE = README.md`).

---

**Author:** dssregi — November 14, 2025

## Quick Start

```bash
# Compile
g++ -std=c++20 -O3 -pthread src/3d_convolution/main.cpp -o demo

# Run
./demo

# Generate documentation
doxygen Doxyfile
open docs/doxygen/html/index.html
```

## Key Features

- **C++20 Work-Stealing Thread Pool** with `std::jthread` and `std::stop_token`
- **Thread-Safe Lock-Free Deques** using `std::atomic` and condition variables
- **Parallel 3D Convolution** with multiple filter kernels (blur, Laplacian, edge detection)
- **Realistic Synthetic Data** with noise for testing filter effectiveness
- **Performance Metrics** and verification output

## Architecture

- **ThreadPool:** Manages worker threads and distributes tasks across queues
- **ThreadSafeDeque:** LIFO (owner) / FIFO (stealing) task queue per thread
- **ConvolutionTask:** Functor encapsulating z-slice convolution work
- **3D Filters:** Gaussian blur, Laplacian, Z-axis edge detection

## Modern C++ Highlights

- `std::jthread` for automatic thread joining
- `std::stop_token` for cooperative cancellation
- `std::unique_ptr` for automatic memory management
- Move semantics and perfect forwarding
- Lock-free atomics for progress tracking
- RAII for exception-safe resource management

## Use Case: Medical Imaging

Processes synthetic 24×24×24 volumetric data (simulating medical scan) with:
- Background tissue (intensity 10)
- Central tumor/feature (intensity 100)
- Realistic Gaussian noise

Applies three types of 3×3×3 convolution filters in parallel for:
- Noise reduction (Gaussian blur)
- Edge/feature detection (Laplacian)
- Directional analysis (Z-axis edge detector)

## Documentation

See the full **Doxygen documentation** for detailed API reference, class hierarchies, and implementation details. After generating HTML:

```bash
doxygen Doxyfile
# Opens to: docs/doxygen/html/index.html
```

---

**Author:** dssregi | **Date:** November 14, 2025

## Credits

For full author and course attribution, see the [CREDITS](CREDITS.md) file.

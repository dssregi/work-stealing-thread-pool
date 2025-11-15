/**
 * @file main.cpp
 * @brief Main entry point for the 3D volumetric filtering demo.
 *
 * This file demonstrates the work-stealing thread pool with 3D convolution tasks.
 * It initializes a synthetic 3D volume containing a central cube with background
 * noise, defines multiple filter kernels (blur, edge detection), and executes
 * them in parallel using the thread pool.
 *
 * @details
 * The program:
 * 1. Creates a ThreadPool with automatic worker thread count.
 * 2. Initializes a 24x24x24 voxel volume with synthetic data and noise.
 * 3. Defines three 3x3x3 convolution kernels:
 *    - Gaussian blur (noise reduction)
 *    - Laplacian (edge/feature detection)
 *    - Z-axis edge detector (directional edge detection)
 * 4. Executes each filter via `execute_convolution`, which submits one task
 *    per z-slice to the thread pool.
 * 5. Prints timing, sample values, and verification metrics.
 * 6. Cleans up via ThreadPool destructor.
 *
 * @author dssregi
 * @version 1.0
 * @date 2025-11-14
 */

#include "convolution.hpp"

/**
 * @brief Main function: initialize pool, data, kernels, and execute filters.
 *
 * @return 0 on successful completion.
 *
 * @details
 * Demonstrates:
 * - Work-stealing thread pool initialization.
 * - Parallel 3D convolution task submission and execution.
 * - Performance metrics and verification output.
 * - Automatic resource cleanup via RAII (destructor on scope exit).
 */
int main() {
    std::cout << "Starting C++20 Parallel 3D Volumetric Filtering (WSD).\n" << std::endl;
    
    // --- 1. Initialization ---
    ThreadPool pool;
    
    Image input_image(VOLUME_SIZE);
    Image output_image(VOLUME_SIZE, 0.0f);
    
    initialize_input_with_cube(input_image);
    
    // --- 2. 3D Kernel Definitions (3x3x3 = 27 elements) ---

    /**
     * @brief Gaussian Blur: uniform 3x3x3 average for noise reduction.
     */
    constexpr float AVG_WEIGHT = 1.0f / (KERNEL_DIM * KERNEL_DIM * KERNEL_DIM); // 1/27
    const std::vector<float> GAUSSIAN_BLUR(27, AVG_WEIGHT); 

    /**
     * @brief Laplacian kernel for edge/feature detection.
     * Center weight=6, neighbor weights=-1. Highlights curvature and edges.
     */
    std::vector<float> LAPLACIAN_KERNEL(27, 0.0f);
    LAPLACIAN_KERNEL[4] = -1.0f; LAPLACIAN_KERNEL[10] = -1.0f; LAPLACIAN_KERNEL[12] = -1.0f; 
    LAPLACIAN_KERNEL[14] = -1.0f; LAPLACIAN_KERNEL[16] = -1.0f; LAPLACIAN_KERNEL[22] = -1.0f; 
    LAPLACIAN_KERNEL[13] = 6.0f; 

    /**
     * @brief Z-axis edge detector: first derivative along the depth axis.
     * Sensitive to depth-wise discontinuities.
     */
    std::vector<float> Z_EDGE_KERNEL(27, 0.0f);
    Z_EDGE_KERNEL[13 + 9] = 1.0f;   
    Z_EDGE_KERNEL[13 - 9] = -1.0f;  

    
    // --- 3. Parallel Execution of Multiple Filters ---

    execute_convolution(pool, input_image, output_image, GAUSSIAN_BLUR, "3D Gaussian Blur (Noise Reduction)");
    execute_convolution(pool, input_image, output_image, LAPLACIAN_KERNEL, "3D Laplacian (Sharpening/Edge)");
    execute_convolution(pool, input_image, output_image, Z_EDGE_KERNEL, "3D Z-Axis Edge Detector");

    std::cout << "\nAll filtering complete. The ThreadPool destructor will now run." << std::endl;
    
    return 0;
}
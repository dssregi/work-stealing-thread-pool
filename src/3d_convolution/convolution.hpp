#ifndef __CONVOLUTION_HPP__
#define __CONVOLUTION_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <atomic>
#include <chrono>
#include <cmath> // For std::sqrt and std::pow
#include <algorithm>
#include <stdexcept>

#include "../core/thread_pool.hpp"

/**
 * @file convolution.hpp
 * @brief 3D volumetric convolution tasks for parallel image filtering.
 *
 * This header provides a complete framework for performing 3D convolution operations
 * on volumetric data (e.g., medical imaging, voxel grids) using a work-stealing
 * thread pool. Tasks are submitted to process depth slices in parallel.
 *
 * @details
 * - The volume is represented as a 1D vector with row-major (C-style) ordering:
 *   index = z * W * H + y * W + x.
 * - Convolution is performed with a 3x3x3 kernel, processing each (y, x) position
 *   across a range of z-slices.
 * - Multiple filter types are defined (Gaussian blur, Laplacian, Z-axis edge).
 * - Results include timing, noise reduction verification, and edge detection metrics.
 *
 * @author dssregi
 * @version 1.0
 * @date 2025-11-14
 */

/**
 * @brief Width of the 3D volume in voxels.
 */
constexpr int IMG_WIDTH = 24;

/**
 * @brief Height of the 3D volume in voxels.
 */
constexpr int IMG_HEIGHT = 24;

/**
 * @brief Depth of the 3D volume in voxels (z-axis).
 */
constexpr int IMG_DEPTH = 24;

/**
 * @brief Kernel dimension (3x3x3 kernel).
 */
constexpr int KERNEL_DIM = 3;

/**
 * @brief Border padding: (KERNEL_DIM - 1) / 2 = 1 voxel on each side.
 */
constexpr int BORDER = KERNEL_DIM / 2;

/**
 * @brief Total number of voxels in the volume.
 */
constexpr int VOLUME_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH;

/**
 * @brief Type alias for 3D volume data.
 *
 * Stored as a 1D vector in row-major order: index = z*W*H + y*W + x.
 */
using Image = std::vector<float>; 

/**
 * @brief Command object (Functor) for executing 3D convolution on depth slices.
 *
 * This class encapsulates a convolution task for a range of depth (z-axis) slices.
 * It implements the Command pattern and is designed to be submitted to the thread pool.
 *
 * @details
 * - Each task processes one or more consecutive z-slices.
 * - For each slice, it iterates over all valid (y, x) positions (excluding borders)
 *   and computes the convolution result using the provided kernel.
 * - Results are written to the output image at the same (z, y, x) position.
 * - An atomic counter is incremented at the end to signal completion.
 *
 * @note
 * The class stores const references to input, kernel, and the output image.
 * Care must be taken to ensure these remain valid during task execution.
 */
class ConvolutionTask {
private:
    /**
     * @brief Const reference to the input 3D volume.
     */
    const Image& input_;

    /**
     * @brief Reference to the output 3D volume where results are written.
     */
    Image& output_;

    /**
     * @brief Const reference to the convolution kernel (27 floats for 3x3x3).
     */
    const std::vector<float>& kernel_;

    /**
     * @brief Starting z-coordinate (depth) of the slice range for this task.
     */
    const int start_slice_;

    /**
     * @brief Ending z-coordinate (exclusive) of the slice range for this task.
     */
    const int end_slice_;

    /**
     * @brief Atomic counter tracking completed slices (for synchronization).
     *
     * Incremented by (end_slice_ - start_slice_) when task completes.
     */
    std::atomic<int>& completed_slices_counter_;

    /**
     * @brief Convert 3D coordinates (z, y, x) to 1D index in row-major order.
     *
     * @param z Z-coordinate (depth)
     * @param y Y-coordinate (row)
     * @param x X-coordinate (column)
     * @return 1D index: z * (W * H) + y * W + x
     */
    inline int get_index(int z, int y, int x) const {
        // z * W * H + y * W + x
        return z * IMG_WIDTH * IMG_HEIGHT + y * IMG_WIDTH + x;
    }

public:
    /**
     * @brief Construct a convolution task for a range of depth slices.
     *
     * @param input The input 3D volume (const reference).
     * @param output The output 3D volume to write results (mutable reference).
     * @param kernel The 3x3x3 convolution kernel (27 floats, const reference).
     * @param start_slice Starting z-coordinate (inclusive).
     * @param end_slice Ending z-coordinate (exclusive).
     * @param completed_slices_counter Atomic counter for synchronization (reference).
     */
    ConvolutionTask(
        const Image& input,
        Image& output,
        const std::vector<float>& kernel,
        int start_slice,
        int end_slice,
        std::atomic<int>& completed_slices_counter)
        : input_(input),
          output_(output),
          kernel_(kernel),
          start_slice_(start_slice),
          end_slice_(end_slice),
          completed_slices_counter_(completed_slices_counter)
    {}

    /**
     * @brief Execute the convolution on the assigned slice range (functor call operator).
     *
     * Iterates over z in [start_slice_, end_slice_) and all valid (y, x) positions,
     * computing the 3D convolution for each output voxel. Updates the completion
     * counter when finished.
     */
    void operator()() const {
        // Loops over the assigned depth slice range (Z-axis)
        for (int z = start_slice_; z < end_slice_; ++z) {
            // Loops over rows (Y-axis) and columns (X-axis)
            for (int r = BORDER; r < IMG_HEIGHT - BORDER; ++r) {
                for (int c = BORDER; c < IMG_WIDTH - BORDER; ++c) {
                    
                    float sum = 0.0f;
                    int kernel_idx = 0;
                    
                    // Iterate over the 3D kernel window (kz, kr, kc)
                    for (int kz = -BORDER; kz <= BORDER; ++kz) {
                        for (int kr = -BORDER; kr <= BORDER; ++kr) {
                            for (int kc = -BORDER; kc <= BORDER; ++kc) {
                                
                                int iz = z + kz;
                                int ir = r + kr;
                                int ic = c + kc;
                                
                                int input_idx = get_index(iz, ir, ic);
                                float k_val = kernel_[kernel_idx++];
                                
                                sum += input_[input_idx] * k_val;
                            }
                        }
                    }
                    
                    // Write the calculated value to the output image
                    output_[get_index(z, r, c)] = sum;
                }
            }
        }
        
        // Signal completion using the atomic counter
        completed_slices_counter_.fetch_add(end_slice_ - start_slice_);
    }
};

/**
 * @brief Initialize the input 3D volume with a central cube and Gaussian noise.
 *
 * @param[out] input The image vector to populate. Must have size >= VOLUME_SIZE.
 *
 * @details
 * Creates a synthetic dataset:
 * - Background set to 10.0 everywhere.
 * - Central cube (5:19, 5:19, 5:19) set to 100.0.
 * - Gaussian noise (mean=0, stddev=8) added to simulate realistic image data.
 */
inline void initialize_input_with_cube(Image& input) {
    // --- 1. Base Data Setup ---
    // Background value
    std::fill(input.begin(), input.end(), 10.0f);
    
    // Define a cube in the center
    constexpr int CUBE_START = 5;
    constexpr int CUBE_END = IMG_DEPTH - CUBE_START; // 19

    for (int z = CUBE_START; z < CUBE_END; ++z) {
        for (int y = CUBE_START; y < CUBE_END; ++y) {
            for (int x = CUBE_START; x < CUBE_END; ++x) {
                // Set the cube's value
                input[z * IMG_WIDTH * IMG_HEIGHT + y * IMG_WIDTH + x] = 100.0f; 
            }
        }
    }

    // --- 2. Add Realistic Gaussian Noise ---
    constexpr float NOISE_MEAN = 0.0f;
    constexpr float NOISE_STDDEV = 8.0f; // Significant noise level to challenge the blur filter

    std::random_device rd;
    // Use a fixed seed or the random device for noise generation
    std::mt19937 generator(rd()); 
    std::normal_distribution<float> distribution(NOISE_MEAN, NOISE_STDDEV);

    for (int i = 0; i < VOLUME_SIZE; ++i) {
        input[i] += distribution(generator);
    }
    
    std::cout << "Input initialized with background (10.0), central cube (100.0), AND Gaussian noise (stdev=" << NOISE_STDDEV << ")." << std::endl;
}

/**
 * @brief Calculate the standard deviation of the background region in the image.
 *
 * @param img The image to analyze.
 * @param label A descriptive label (printed in output).
 * @return The standard deviation of the sampled region.
 *
 * @details
 * Samples the background region (first few slices excluding borders) to estimate
 * noise levels. Useful for verifying noise reduction filters.
 */
inline float calculate_std_dev(const Image& img, const std::string& label) {
    // Sample a uniform region: the background in the first few slices (excluding borders)
    constexpr int SAMPLE_Z_END = 5; 
    std::vector<float> sample;
    int index_count = 0;
    
    for (int z = BORDER; z < SAMPLE_Z_END; ++z) {
        for (int r = BORDER; r < IMG_HEIGHT - BORDER; ++r) {
            for (int c = BORDER; c < IMG_WIDTH - BORDER; ++c) {
                // The cube starts at Z=5, so this captures only background noise
                sample.push_back(img[z * IMG_WIDTH * IMG_HEIGHT + r * IMG_WIDTH + c]);
                index_count++;
            }
        }
    }

    if (sample.empty()) return 0.0f;

    // Calculate Mean
    double sum = 0.0;
    for (float val : sample) {
        sum += val;
    }
    float mean = sum / index_count;

    // Calculate Variance
    double variance_sum = 0.0;
    for (float val : sample) {
        variance_sum += std::pow(val - mean, 2);
    }
    float variance = variance_sum / (index_count - 1); // Use (N-1) for sample standard deviation

    // Calculate Standard Deviation
    float std_dev = std::sqrt(variance);

    std::cout << "   " << label << " (Background region): Std Dev = " << std_dev << std::endl;
    return std_dev;
}

/**
 * @brief Execute 3D convolution with a specified kernel using the thread pool.
 *
 * @param pool Reference to the ThreadPool for parallel execution.
 * @param input The input 3D volume (const reference).
 * @param[out] output The output 3D volume (mutable reference, will be zeroed).
 * @param kernel The convolution kernel: 27 floats for 3x3x3 (const reference).
 * @param kernel_name Descriptive name of the kernel (for logging).
 *
 * @details
 * - Submits one task per z-slice to the thread pool for parallel processing.
 * - Blocks until all tasks complete (monitored via atomic counter).
 * - Logs timing information, center, and edge voxel values for verification.
 * - Commented verification code allows deeper analysis of filter effects.
 *
 * @note
 * The output buffer is reset to zero before processing. This function blocks
 * the caller until all convolution tasks complete.
 */
inline void execute_convolution(ThreadPool& pool, const Image& input, Image& output, 
                         const std::vector<float>& kernel, const std::string& kernel_name) 
{
    using namespace std::literals;

    // Reset output image to zero before each filter run
    std::fill(output.begin(), output.end(), 0.0f);
    std::atomic<int> completed_slices = 0;
    int processable_slices = IMG_DEPTH - 2 * BORDER;
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // Iterate over the depth axis (Z) and submit one task per slice
    for (int z = BORDER; z < IMG_DEPTH - BORDER; ++z) {
        ConvolutionTask task(
            input, 
            output, 
            kernel, 
            z,          // start_slice
            z + 1,      // end_slice (processing one slice at a time)
            completed_slices
        );
        
        // Use a lambda to wrap the functor for submission to the ThreadPool
        pool.submit([task](){ task(); });
    }

    std::cout << "\n[Filter: " << kernel_name << "] Submitted " << processable_slices << " tasks." << std::endl;

    // Wait for Completion 
    while (completed_slices.load() < processable_slices) {
        std::this_thread::sleep_for(1ms); 
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Time taken for parallel processing: " << duration.count() << " ms" << std::endl;

    // --- VERIFICATION ---
    
    // Coordinates for central voxel (inside cube, uniform)
    int center_z = IMG_DEPTH / 2; // 12
    int center_y = IMG_HEIGHT / 2; // 12
    int center_x = IMG_WIDTH / 2; // 12
    int center_idx = center_z * IMG_WIDTH * IMG_HEIGHT + center_y * IMG_WIDTH + center_x;
    float center_value = output[center_idx];

    // Coordinates for edge voxel (right on the boundary of the cube, high contrast)
    int edge_z = 5; 
    int edge_y = IMG_HEIGHT / 2; 
    int edge_x = IMG_WIDTH / 2; 
    int edge_idx = edge_z * IMG_WIDTH * IMG_HEIGHT + edge_y * IMG_WIDTH + edge_x;
    float edge_value = output[edge_idx];

    // Note: Detailed verification code is commented out for brevity.
    // if (kernel_name.find("Gaussian Blur") != std::string::npos) {
    //     // Verification for Noise Reduction
    //     float input_std_dev = calculate_std_dev(input, "Input Noise (high)");
    //     float output_std_dev = calculate_std_dev(output, "Output Noise (low)");
    //     std::cout << "VERIFIED: Noise reduction factor (Input/Output StdDev): " << input_std_dev / output_std_dev << std::endl;
    //     std::cout << "Result: Center Voxel value (should be ~100.0): " << center_value << std::endl;
    // } else if (kernel_name.find("Laplacian") != std::string::npos) {
    //     // Verification for Edge Detection
    //     std::cout << "VERIFIED: Laplacian filter functionality (Should be near 0 in uniform areas, high on edges)." << std::endl;
    //     std::cout << "Result: Center Voxel value (should be ~0.0): " << center_value << std::endl;
    //     // The edge value should be high due to the 90.0 contrast jump (100.0 - 10.0)
    //     std::cout << "Result: Edge Voxel value (should be high spike): " << edge_value << std::endl;
    // } else if (kernel_name.find("Z-Axis Edge Detector") != std::string::npos) {
    //     // Verification for Directional Edge
    //     std::cout << "VERIFIED: Z-Edge filter functionality." << std::endl;
    //     std::cout << "Result: Center Voxel value (should be ~0.0): " << center_value << std::endl;
    //     // Checking Z=4 (background): Z+1(cube edge) - Z-1(background) = 100 - 10 = 90 (plus noise effects).
    //     std::cout << "Result: Edge Voxel at Z=4 (should be high spike): " << output[4 * IMG_WIDTH * IMG_HEIGHT + edge_y * IMG_WIDTH + edge_x] << std::endl;
    // }
}

#endif // __CONVOLUTION_HPP__
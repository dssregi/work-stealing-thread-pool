#ifndef __THREAD_POOL_HPP__
#define __THREAD_POOL_HPP__

#include <thread>
#include <vector>
#include <functional>
#include <memory>
#include <random>
#include <mutex>
#include <stop_token> 
#include <algorithm> 
#include <iostream>

#include "thread_safe_deque.hpp"

/**
 * @file thread_pool.hpp
 * @brief C++20 work-stealing thread pool implementation.
 *
 * This header implements a thread pool using work-stealing queue semantics with C++20
 * structured concurrency (`std::jthread` and `std::stop_token`). Each worker thread
 * has its own deque and performs work-stealing from randomly selected peer queues
 * when its local queue is empty.
 *
 * @details
 * - Worker threads are created at construction time based on hardware concurrency.
 * - Tasks are submitted to randomly selected queues to achieve load distribution.
 * - Each thread preferentially executes from its own queue (LIFO), then steals
 *   from peers (FIFO) to improve cache locality and work distribution.
 * - Graceful shutdown is triggered when the destructor is called, with all
 *   pending tasks executed before thread join.
 *
 * @author dssregi
 * @version 1.0
 * @date 2025-11-14
 */

/**
 * @brief Function type alias for tasks submitted to the thread pool.
 *
 * Tasks are void-returning, parameterless callables (e.g., lambdas, std::function).
 */
using TaskFunc = std::function<void()>;

/**
 * @brief Queue type alias for thread-safe work-stealing deques.
 *
 * Each thread in the pool owns one such queue to hold its tasks.
 */
using Queue = ThreadSafeDeque<TaskFunc>;

/**
 * @brief Work-stealing thread pool for parallel task execution.
 *
 * @details
 * The pool maintains a deque of queues (one per worker thread). Tasks are submitted
 * to a random queue. Worker threads execute work from their own queue in LIFO order
 * (improving cache locality), and steal from peers' queues in FIFO order when idle.
 *
 * The pool uses C++20 features:
 * - `std::jthread` for automatic thread management and joining.
 * - `std::stop_token` for cooperative cancellation.
 *
 * @thread_safety Thread pool operations are safe for concurrent task submission.
 *                Shutdown is coordinated via `stop_source_` and condition variables
 *                in the underlying `ThreadSafeDeque`.
 */
class ThreadPool {
private:
    /**
     * @brief Stop source for signalling worker threads to exit via stop tokens.
     */
    std::stop_source stop_source_;

    /**
     * @brief Vector of worker threads managed via C++20 jthreads.
     *
     * jthreads automatically join when destroyed, providing RAII semantics.
     */
    std::vector<std::jthread> threads;

    /**
     * @brief Array of work-stealing deques, one per worker thread.
     *
     * Tasks are submitted to random queues and stolen across queues for load balancing.
     */
    std::unique_ptr<Queue[]> work_queues;

    /**
     * @brief Mersenne Twister RNG for random queue selection during submission and stealing.
     */
    std::mt19937 mt;

    /**
     * @brief Mutex protecting the RNG to ensure thread-safe random generation.
     */
    std::mutex rand_mut;

    /**
     * @brief Number of worker threads in this pool.
     */
    int thread_count;

    /**
     * @brief Worker thread entry point.
     *
     * @param token Stop token for cooperative cancellation.
     * @param idx Zero-based index of this worker thread.
     *
     * @details
     * Executes the work-stealing loop:
     *   1. Try LIFO pop from own queue (cache-friendly).
     *   2. Try FIFO steal from a random peer queue.
     *   3. Block on own queue until task available or close() called.
     */
    void worker(std::stop_token token, int idx);

    /**
     * @brief Generate a random queue index uniformly in [0, thread_count).
     *
     * @return Random queue index.
     */
    int get_random();

    /**
     * @brief Close all worker queues to trigger shutdown.
     *
     * Called during destruction to signal blocked `wait_and_pop` calls.
     */
    void stop_workers(); 

public:
    /**
     * @brief Construct a ThreadPool with worker threads.
     *
     * Initializes worker threads based on hardware concurrency (capped at 4 for demo).
     * Seeds the RNG and creates work queues for each thread.
     */
    ThreadPool();

    /**
     * @brief Destroy the ThreadPool and wait for all workers to finish.
     *
     * Requests stop via `stop_source_`, closes all queues, and joins all jthreads.
     */
    ~ThreadPool();

    /**
     * @brief Disable copy construction.
     */
    ThreadPool(const ThreadPool&) = delete;

    /**
     * @brief Disable copy assignment.
     */
    ThreadPool& operator =(const ThreadPool&) = delete;

    /**
     * @brief Submit a task to the thread pool for execution.
     *
     * The task is added to a randomly selected work queue. It will be executed
     * by a worker thread at some point during the pool's lifetime.
     *
     * @param func Callable task to execute (must be convertible to `TaskFunc`).
     */
    void submit(TaskFunc func);
};

/**
 * @details
 * @name Inline Implementation of ThreadPool methods
 * @{
 */

/**
 * @brief Constructor implementation: initialize threads and queues.
 */
inline ThreadPool::ThreadPool() {
    thread_count = std::max(1, (int)std::thread::hardware_concurrency());
    std::cout << "ThreadPool starting with " << thread_count << " worker threads." << std::endl;

    std::random_device rd;
    mt.seed(rd());

    work_queues = std::make_unique<Queue[]>(thread_count);

    for (int i = 0; i < thread_count; ++i) {
        threads.emplace_back([this, i](std::stop_token token) {
            this->worker(std::move(token), i);
        });
    }
}

/**
 * @brief Destructor implementation: request stop and join all threads.
 */
inline ThreadPool::~ThreadPool() {
    stop_source_.request_stop(); 
    stop_workers();
    std::cout << "ThreadPool shutting down cleanly. All jthreads joined." << std::endl;
}

/**
 * @brief Implementation of stop_workers: close all queues to signal exit.
 */
inline void ThreadPool::stop_workers() {
    for (int i = 0; i < thread_count; ++i) {
        work_queues[i].close();
    }
}

/**
 * @brief Implementation of worker: main loop for work-stealing execution.
 */
inline void ThreadPool::worker(std::stop_token token, int idx) {
    TaskFunc task;
    
    while (!token.stop_requested()) { 
        // 1. Primary: Try LIFO pop from own queue (optimal cache use)
        if (work_queues[idx].try_pop(task)) {
            task();
            continue;
        }

        // 2. Stealing: Check a random queue
        int i = get_random();
        
        // Use try_steal (FIFO pop) from the random queue
        if (work_queues[i].try_steal(task)) { 
            task();
            continue;
        }
        
        // 3. Last Resort: Block efficiently on our own queue (LIFO pop)
        // If wait_and_pop returns false, it means close() was called and the queue is empty.
        if (!work_queues[idx].wait_and_pop(task)) {
            break; 
        }
        
        if (task) {
            task();
        }
    }
    std::cout << "Worker " << idx << " exited." << std::endl;
}

/**
 * @brief Implementation of get_random: thread-safe RNG for queue selection.
 */
inline int ThreadPool::get_random() {
    std::lock_guard<std::mutex> lck_guard(rand_mut);
    std::uniform_int_distribution<int> dist(0, thread_count - 1);
    return dist(mt);
}

/**
 * @brief Implementation of submit: push task to random queue.
 */
inline void ThreadPool::submit(TaskFunc func) {
    int i = get_random();
    work_queues[i].push(std::move(func)); 
}

/**
 * @}
 */

#endif // __THREAD_POOL_HPP__
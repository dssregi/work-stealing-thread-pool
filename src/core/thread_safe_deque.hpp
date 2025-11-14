#ifndef __THREAD_SAFE_DEQUE_HPP__
#define __THREAD_SAFE_DEQUE_HPP__

#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>

using namespace std::literals;

/**
 * @file thread_safe_deque.hpp
 * @brief Thread-safe work-stealing deque declaration.
 *
 * This header provides a small thread-safe work-stealing deque designed for
 * a work-stealing thread-pool. The owner of the deque performs LIFO
 * operations (push/pop at the back) while other threads may "steal" work
 * from the front (FIFO). The implementation internally uses
 * `std::deque<std::unique_ptr<T>>` to efficiently move and manage task objects.
 *
 * @note This file has no external dependencies beyond the standard library.
 *
 * @author dssregi
 * @version 1.0
 * @date 2025-11-14
 */

/**
 * @brief Thread-safe work-stealing deque template.
 *
 * @tparam T Type of the objects stored in the deque. Objects are stored as
 *           `std::unique_ptr<T>` internally so `T` must be MoveConstructible.
 *
 * @details
 * - Owner threads should push and pop from the back (LIFO) to benefit from
 *   cache locality.
 * - Stealing threads should call `try_steal` which pops from the front (FIFO)
 *   to obtain older tasks.
 * - Blocking behavior is provided through `push` (blocks when full) and
 *   `wait_and_pop` (blocks until not empty or closed). `try_pop` and
 *   `try_steal` are non-blocking.
 *
 * @thread_safety The class is safe for concurrent use: multiple threads may
 *                call stealing methods while a single owner thread performs
 *                owner operations. Internal synchronization is implemented
 *                with `std::mutex` and `std::condition_variable`.
 */
template <class T>
class ThreadSafeDeque {
private:
    /**
     * @brief Mutex protecting the internal deque and condition variables.
     */
    std::mutex mut_;

    /**
     * @brief Container holding the tasks as owning pointers.
     *
     * Using `std::unique_ptr<T>` avoids copies and clearly transfers
     * ownership when elements are popped or stolen.
     */
    std::deque<std::unique_ptr<T>> deque_;

    /**
     * @brief Maximum number of elements allowed in the deque before `push`
     *        blocks.
     */
    const size_t max_size_;
    
    /**
     * @brief Condition variable signalled when the deque becomes non-empty.
     */
    std::condition_variable cv_not_empty_;

    /**
     * @brief Condition variable signalled when the deque has space for pushes.
     */
    std::condition_variable cv_not_full_;
    
    /**
     * @brief When true, the deque is closed and blocking waits should return.
     */
    bool done_ = false;

public:
    /**
     * @brief Construct a ThreadSafeDeque with a maximum capacity.
     *
     * @param max_size Maximum number of entries before `push` blocks.
     */
    ThreadSafeDeque(size_t max_size = 50) : max_size_(max_size) {}
    
    /**
     * @brief Disable copy construction.
     */
    ThreadSafeDeque(const ThreadSafeDeque&) = delete;

    /**
     * @brief Disable copy assignment.
     */
    ThreadSafeDeque& operator =(const ThreadSafeDeque&) = delete;

    /**
     * @brief Push a new value onto the back of the deque (owner operation).
     *
     * This call will block if the deque has reached `max_size_` until space
     * becomes available or `close()` is called.
     *
     * @param value The value to push. It will be moved into the container.
     *
     * @note If `close()` has been called, `push` will return immediately and
     *       the value will not be stored.
     */
    void push(T value) {
        std::unique_ptr<T> data_ptr = std::make_unique<T>(std::move(value));

        std::unique_lock<std::mutex> lock(mut_);
        cv_not_full_.wait(lock, [this]{ return done_ || deque_.size() < max_size_; });

        if (done_) { 
            return; 
        }

        deque_.push_back(std::move(data_ptr)); // LIFO Push to back
        cv_not_empty_.notify_one();
    }

    /**
     * @brief Try to pop an element from the back (owner LIFO pop) without blocking.
     *
     * @param[out] value Where the popped value is placed if pop succeeds.
     * @return true if an element was popped, false if the deque was empty.
     */
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mut_);
        
        if (deque_.empty()) {
            return false;
        }
        
        // LIFO Pop from back (improves cache locality for the owner)
        std::unique_ptr<T> data_ptr = std::move(deque_.back());
        deque_.pop_back();
        
        value = std::move(*data_ptr); 
        cv_not_full_.notify_one(); 
        return true;
    }
    
    /**
     * @brief Try to steal an element from the front (non-owner FIFO pop) without blocking.
     *
     * Stealing threads should use this method to obtain older work items.
     *
     * @param[out] value Where the stolen value is placed if steal succeeds.
     * @return true if an element was stolen, false if the deque was empty.
     */
    bool try_steal(T& value) {
        std::lock_guard<std::mutex> lock(mut_);
        
        if (deque_.empty()) {
            return false;
        }

        // FIFO Pop from front (stealing the oldest work)
        std::unique_ptr<T> data_ptr = std::move(deque_.front());
        deque_.pop_front();

        value = std::move(*data_ptr);
        cv_not_full_.notify_one();
        return true;
    }

    /**
     * @brief Wait until an element is available and pop it from the back (owner LIFO pop).
     *
     * This method blocks until the deque is not empty or `close()` is called.
     * If `close()` has been called and the deque is empty, this returns false.
     *
     * @param[out] value Where the popped value is placed if pop succeeds.
     * @return true if an element was popped, false if the deque was closed and empty.
     */
    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mut_);
        
        cv_not_empty_.wait(lock, [this]{ return done_ || !deque_.empty(); });

        if (done_ && deque_.empty()) {
            return false; 
        }

        // LIFO Pop from back
        std::unique_ptr<T> data_ptr = std::move(deque_.back());
        deque_.pop_back();

        value = std::move(*data_ptr);
        cv_not_full_.notify_one();
        return true;
    }

    /**
     * @brief Close the deque and wake any blocking waiters.
     *
     * After calling `close()`, blocked `push` or `wait_and_pop` calls will
     * return (push will no-op, wait_and_pop will return false if empty).
     */
    void close() {
        std::lock_guard<std::mutex> lock(mut_);
        done_ = true;
        cv_not_empty_.notify_all(); 
        cv_not_full_.notify_all();  
    }
};

#endif // __THREAD_SAFE_DEQUE_HPP__
/**
 * @file
 * @brief Contains functions that create and interact with threadpool
 * @author Raja Nand Sharma
 */

#ifndef __THREADPOOL
#define __THREADPOOL

#include <pthread.h>

/**
 * @struct _threadpool
 * @brief Defines a threadpool struct
 */
typedef struct {
    pthread_t *threads;  ///< Stores the threads
    int nthread;         ///< Stores the number of threads
} threadpool_t;

/**
 * @brief Creates a threadpool
 */
threadpool_t *create_threadpool();

/**
 * @brief Adds a thread to a threadpool
 * @param pool Pointer to a threadpool
 * @param thread pthread_t to add to the threadpool
 */
void submit_thread(threadpool_t *pool, pthread_t thread);

/**
 * @brief Waits for all the threads in the pool to finish execution
 * @param pool Pointer to a threadpool
 */
void threadpool_wait(threadpool_t *pool);

/**
 * @brief Destroys a threadpool
 * @param pool Pointer to a threadpool
 */
void destroy_threadpool(threadpool_t *pool);

#endif
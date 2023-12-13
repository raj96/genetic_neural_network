#include <stdlib.h>
#include <util/threadpool.h>

threadpool_t *create_threadpool() {
    threadpool_t *threadpool = (threadpool_t *)malloc(sizeof(threadpool_t));
    threadpool->nthread = 0;
    threadpool->threads = NULL;

    return threadpool;
}

void submit_thread(threadpool_t *pool, pthread_t thread) {
    pool->nthread++;
    pool->threads = (pthread_t *)realloc(pool->threads, sizeof(pthread_t) * pool->nthread);
    pool->threads[pool->nthread - 1] = thread;
}

void threadpool_wait(threadpool_t *pool) {
    for (int i = 0; i < pool->nthread; i++) {
        pthread_join(pool->threads[i], NULL);
    }
}

void destroy_threadpool(threadpool_t *pool) {
    free(pool);
}
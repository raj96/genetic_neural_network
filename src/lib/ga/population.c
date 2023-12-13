#include <ga/population.h>
#include <util/threadpool.h>

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

ga_population *create_population(int population_size, float mutation_rate, fitness_fx f_fx, populate_fx p_fx) {
    ga_population *population = (ga_population *)malloc(sizeof(ga_population));

    population->population_size = population_size;
    population->mutation_rate = mutation_rate;
    population->f_fx = f_fx;
    population->fittest = 0;

    p_fx(population);

    return population;
}

typedef struct {
    ga_population *population;
    int index;
    pthread_mutex_t *lock;          //         <-------------+
    float *best_fitness_score;      // Protected by lock ----+
} __population_fitness_threadtask_params;

void* __population_fitness_threadtask(void *params) {
    __population_fitness_threadtask_params *p = (__population_fitness_threadtask_params *)params;

    float fitness_score = p->population->f_fx(p->population->population[p->index]);
    pthread_mutex_lock(p->lock);
    if (fitness_score > *(p->best_fitness_score)) {
        *(p->best_fitness_score) = fitness_score;
        p->population->fittest = p->index;
    }
    pthread_mutex_unlock(p->lock);
}

float __run_fitness_over_population(ga_population *population) {
    pthread_mutex_t lock;
    float best_fitness_score = 0;

    if (pthread_mutex_init(&lock, NULL) != 0) {
        printf("Mutex init failed");
        exit(1);
    }

    threadpool_t *pool = create_threadpool();
    for (int i = 0; i < population->population_size; i++) {
        pthread_t thread;
        __population_fitness_threadtask_params *params = (__population_fitness_threadtask_params *)malloc(sizeof(__population_fitness_threadtask_params));

        params->population = population;
        params->index = i;
        params->lock = &lock;
        params->best_fitness_score = &best_fitness_score;

        pthread_create(&thread, NULL, &__population_fitness_threadtask, params);
        submit_thread(pool, thread);
    }

    threadpool_wait(pool);
    destroy_threadpool(pool);

    return best_fitness_score;
}

void population_run_ga_for_generation(ga_population *population, int ngen, crossover_fx c_fx, runner_fx r_fx) {
    while (ngen > 0) {
        float best_fitness_score = __run_fitness_over_population(population);
        c_fx(population);
        if (r_fx != NULL) r_fx(population, best_fitness_score);

        ngen--;
    }
}

void population_run_ga_until_fitness_over(ga_population *population, float fitness_barrier, crossover_fx c_fx, runner_fx r_fx) {
    float best_fitness_score = 0;
    do {
        best_fitness_score = __run_fitness_over_population(population);
        c_fx(population);
        if (r_fx != NULL) r_fx(population, best_fitness_score);

    } while (best_fitness_score < fitness_barrier);
}

void destroy_population(ga_population *population, destroy_members_fx destroy_fx) {
    destroy_fx(population);

    free(population);
}

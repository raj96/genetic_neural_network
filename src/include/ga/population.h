/**
 * @file
 * @brief Contains structs and functions related to population
 * @author Raja Nand Sharma
 */

#ifndef __POPULATION
#define __POPULATION

typedef float (*fitness_fx)(void *);

/**
 * @struct ga_population
 * @brief Defines a population structure for GA
 */
typedef struct
{
    void **population;    ///< Population, the type of the variable will be dictated by p_fx function
    int population_size;  ///< Size of the population
    float mutation_rate;  ///< Mutation rate for the genetic algorithm
    fitness_fx f_fx;      ///< Fitness function for the genetic algorithm
    int fittest;          ///< Contains the index of the fittest member of the population
} ga_population;

typedef void (*crossover_fx)(ga_population *);
typedef void (*populate_fx)(ga_population *);
typedef void (*runner_fx)(ga_population *, float);
typedef void (*destroy_members_fx)(ga_population *);

/**
 * @brief Creates and returns a pointer to the population
 * @param population_size Size of the population in a generation
 * @param mutation_rate Mutation rate for every element in the population
 * @param f_fx Fitness function, takes in a void * (this will be a single element from the population member of the ga_population struct), the fitness should be calculated and a float should be returned denoting the fitness
 * @param p_fx Populate function, takes in a ga_population * variable and should populate the population member of the struct
 */
ga_population *create_population(int population_size, float mutation_rate, fitness_fx f_fx, populate_fx p_fx);

/**
 * @brief Runs genetic algorithm for a given number of generation
 * @param population Pointer to a ga_population struct to run the algorithm on
 * @param ngen Number of generations to run the algorithm for
 * @param c_fx Crossover function for the genetic algorithm, this function should re-populate the populations as well
 * @param r_fx Pointer to a function that runs after every generation, with a pointer to the current ga_population and fitness score of the fittest candidate for the prior generation passed to it. NOTE: The next generation won't commence untile r_fx returns. This field can be set to NULL as well.
 *
 */
void population_run_ga_for_generation(ga_population *population, int ngen, crossover_fx c_fx, runner_fx r_fx);

/**
 * @brief Runs genetic algorithm until the fitness value for atleast one member of the population is equal to or less than the fitness barrier
 * @param population Pointer to a ga_population struct to run the algorithm on
 * @param fitness_barrier Maximum fitness value, set this to FLT_MAX to run the algorithm forever
 * @param c_fx Crossover function for the genetic algorithm, this function should re-populate the populations as well
 * @param r_fx Pointer to a function that runs after every generation, with a pointer to the current ga_population and fitness score of the fittest candidate for the prior generation passed to it. NOTE: The next generation won't commence untile r_fx returns. This field can be set to NULL as well.
 */
void population_run_ga_until_fitness_over(ga_population *population, float fitness_barrier, crossover_fx c_fx, runner_fx r_fx);

/**
 * @brief Destroys a ga_population struct
 * @param population Pointer to ga_population struct to be destroyed
 * @param destroy_fx Pointer to a function that will deallocate resources indside ga_population that were allocated when the population was created. DO NOT free the ga_population struct
*/
void destroy_population(ga_population *population, destroy_members_fx destroy_fx);

#endif

#include <ga/population.h>
#include <math.h>
#include <nn/net.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define E exp(1)

typedef struct
{
    int training_data_len;
    int input_dim;
    int output_dim;
    float **inputs;
    float **outputs;
} training_data_t;

static training_data_t training_data = {
    .input_dim = 2,
    .output_dim = 1,
    .training_data_len = 4,
};

void populate_training_data() {
    training_data.inputs = (float **)malloc(sizeof(float) * training_data.training_data_len);
    training_data.outputs = (float **)malloc(sizeof(float) * training_data.training_data_len);
    for (int i = 0; i < training_data.training_data_len; i++) {
        training_data.inputs[i] = (float *)malloc(sizeof(float) * training_data.input_dim);
        training_data.outputs[i] = (float *)malloc(sizeof(float) * training_data.output_dim);
    }

    training_data.inputs[0][0] = 0.0f;
    training_data.inputs[0][1] = 0.0f;
    training_data.outputs[0][0] = 0.0f;

    training_data.inputs[1][0] = 0.0f;
    training_data.inputs[1][1] = 1.0f;
    training_data.outputs[1][0] = 1.0f;

    training_data.inputs[2][0] = 1.0f;
    training_data.inputs[2][1] = 0.0f;
    training_data.outputs[2][0] = 1.0f;

    training_data.inputs[3][0] = 1.0f;
    training_data.inputs[3][1] = 1.0f;
    training_data.outputs[3][0] = 0.0f;
}

void destroy_training_data() {
    for (int i = 0; i < training_data.training_data_len; i++) {
        free(training_data.inputs[i]);
        free(training_data.outputs[i]);
    }

    free(training_data.inputs);
    free(training_data.outputs);
}

float relu(float x) {
    return x < 0 ? 0 : x;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + pow(E, -x));
}

float fitness_f(void *member) {
    nn_net *net = (nn_net *)member;
    float fitness = 0.0f;

    for (int i = 0; i < training_data.training_data_len; i++) {
        float *predicted_output = forward_propagate_net(net, training_data.inputs[i]);
	for (int j = 0; j < training_data.output_dim; j++) {
            float unfit_val = pow(training_data.outputs[i][j] - predicted_output[j], 2);
            // High Penalty Low Reward
            unfit_val = unfit_val > 0.1 ? (1/unfit_val) * 10 : ( unfit_val == 0 ? 10e-25  : 1/unfit_val);

	    fitness += unfit_val; 
        }
    }

    return fitness;
}

void populate(ga_population *population) {
    population->population = (void **)malloc(sizeof(nn_net) * population->population_size);

    for (int i = 0; i < population->population_size; i++) {
        nn_net *net = create_net();
        net_add_layer(net, training_data.input_dim, relu);
        net_add_layer(net, 4, sinf);
        net_add_layer(net, 3, sigmoid);
        net_add_layer(net, training_data.output_dim, sigmoid);

        population->population[i] = (void *)net;
    }
}

void crossover_f(ga_population *population) {
    nn_net *fittest_net = (nn_net *)population->population[population->fittest];

    for (int i = 0; i < population->population_size; i++) {
        nn_net *net = (nn_net *)population->population[i];
	if(i == population->fittest) continue;

        for (int j = ((float)rand() / RAND_MAX) < 0.5 ? 0 : 1; j < net->n_layer_weights; j+=2) {
            // net->layer_weights[j] *= 0.5;
            // net->layer_weights[j] += 0.5*fittest_net->layer_weights[j];
       	    net->layer_weights[j] = fittest_net->layer_weights[j];
	    if (((float)rand() / RAND_MAX) < population->mutation_rate) {
	        net->layer_weights[j] = (float)rand() / RAND_MAX;
	    }
	}
    }
}

void destroy_f(ga_population *population) {
    for (int i = 0; i < population->population_size; i++) {
        // destroy_net((nn_net *)population->population[i]);
    }

    free(population->population);
}

void iter_f(ga_population *population, float best_score) {
	nn_net *fittest = population->population[population->fittest];
	printf("Fittest spits(%f): ", best_score);

	for(int i = 0; i < training_data.training_data_len; i++) {
		float *output = forward_propagate_net(fittest, training_data.inputs[i]);

		printf("\t(%.1g,%.1g)=>%.5g", training_data.inputs[i][0], training_data.inputs[i][1], output[0]);
	}
	printf("\n");
}

int main() {
    srand(time(NULL));
    populate_training_data();
    ga_population *population = create_population(200, 1, &fitness_f, &populate);

    // for(int i = 0; i < population->population_size; i++) {
    //     nn_net *net = population->population[i];
    //     printf("Net %d weights(%u):\t[", i, net);
    //     for (int i = 0; i < net->n_layer_weights; i++) {
    //         printf("%f ", net->layer_weights[i]);
    //     }
    //     printf("]\n");
    // }

    population_run_ga_for_generation(population, 0.5, &crossover_f, NULL);

    nn_net *fittest = population->population[population->fittest];
    for (int i = 0; i < training_data.training_data_len; i++) {
        printf("%f %f:\t%f\n", training_data.inputs[i][0], training_data.inputs[i][1], forward_propagate_net(fittest, training_data.inputs[i])[0]);
    }
/*    for (int i = 0; i < fittest->n_layer_weights; i++) {
        printf("%f ", fittest->layer_weights[i]);
    }
*/    printf("\nFittest: %d\n", population->fittest);

    //population_run_ga_for_generation(population, 100, &crossover_f, &iter_f);
    population_run_ga_until_fitness_over(population, 1000, &crossover_f, &iter_f);

    fittest = population->population[population->fittest];
    for (int i = 0; i < training_data.training_data_len; i++) {
        printf("%f %f:\t%f\n", training_data.inputs[i][0], training_data.inputs[i][1], forward_propagate_net(fittest, training_data.inputs[i])[0]);
    }
/*    for (int i = 0; i < fittest->n_layer_weights; i++) {
        printf("%f ", fittest->layer_weights[i]);
    }
*/    printf("\nFittest: %d\n", population->fittest);

    destroy_population(population, &destroy_f);
    // destroy_training_data();

    return 0;
}

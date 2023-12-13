#include <nn/layer.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <util/threadpool.h>

nn_layer *create_layer(int n_nodes, int n_prev_nodes, activation_fx a_fx, float **layer_weights, int *n_layer_weights) {
    nn_layer *new_layer = (nn_layer *)malloc(sizeof(nn_layer));

    new_layer->_nweights = n_nodes * n_prev_nodes;
    new_layer->_nnodes = n_nodes;

    new_layer->nodes = (float *)malloc(sizeof(float) * new_layer->_nnodes);
    new_layer->a_fx = a_fx;

    new_layer->w_i_start = *n_layer_weights;

    *layer_weights = (float *)realloc(*layer_weights, sizeof(float) * (*n_layer_weights + new_layer->_nweights));
    *n_layer_weights += new_layer->_nweights;

    int n;
    for (n = new_layer->w_i_start; n < (new_layer->w_i_start + new_layer->_nweights); n++) {
        *(*layer_weights + n) = (float)rand() / RAND_MAX;
    }

    return new_layer;
}

void destroy_layer(nn_layer *layer) {
    free(layer->nodes);
    free(layer);
}

typedef struct
{
    float *node;
    float *prev_nodes;
    int current_node_i;
    int prev_nodes_length;
    int start_index;
    float *layer_weights;
    activation_fx a_fx;
} __update_layer_weight_params;

void *__update_layer_weight(void *_params) {
    __update_layer_weight_params params = *((__update_layer_weight_params *)_params);
    int w_index = params.start_index + (params.current_node_i * params.prev_nodes_length);
    *(params.node) = 0.0f;
    for (int j = 0; j < params.prev_nodes_length; j++) {
        *(params.node) += params.layer_weights[w_index] * *(params.prev_nodes + j);
        w_index++;
    }
    *(params.node) = params.a_fx(*params.node);

    return NULL;
}

void forward_propagate_layer(nn_layer *layer1, nn_layer *layer2, float *layer_weights) {
    threadpool_t *tp = create_threadpool();

    for (int i = 0; i < layer2->_nnodes; i++) {
        pthread_t thread;
        __update_layer_weight_params *params = (__update_layer_weight_params *)malloc(sizeof(__update_layer_weight_params));

        params->node = &(layer2->nodes[i]);
        params->current_node_i = i;
        params->prev_nodes = layer1->nodes;
        params->prev_nodes_length = layer1->_nnodes;
        params->start_index = layer2->w_i_start;
        params->layer_weights = layer_weights;
        params->a_fx = layer2->a_fx;

        pthread_create(&thread, NULL, &__update_layer_weight, params);
        submit_thread(tp, thread);
    }

    threadpool_wait(tp);
    destroy_threadpool(tp);
}

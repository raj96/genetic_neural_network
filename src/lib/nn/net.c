#include <nn/net.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>



nn_net *create_net() {
    nn_net *net = (nn_net *)malloc(sizeof(nn_net));

    net->layers = NULL;
    net->n_layers = 0;
    net->layer_weights = NULL;
    net->n_layer_weights = 0;

    // srand(time(NULL));

    return net;
}

void net_add_layer(nn_net *net, int nnodes, activation_fx a_fx) {
    int prev_nnodes = net->n_layers > 0 ? net->layers[net->n_layers - 1]->_nnodes : 0;
    nn_layer *layer = create_layer(nnodes, prev_nnodes, a_fx, &net->layer_weights, &net->n_layer_weights);

    net->layers = (nn_layer **)realloc(net->layers, sizeof(nn_layer) * net->n_layers + 1);
    net->layers[net->n_layers++] = layer;
}

void destroy_net(nn_net *net) {
    for (int i = 0; i < net->n_layers; i++) {
        free(net->layers[i]);
    }
    free(net->layers);
    free(net->layer_weights);

    free(net);
}

float *forward_propagate_net(nn_net *net, float *input) {
    for (int i = 0; i < net->layers[0]->_nnodes; i++) {
        net->layers[0]->nodes[i] = input[i];
    }

    for (int i = 0; i < net->n_layers - 1; i++) {
        forward_propagate_layer(net->layers[i], net->layers[i + 1], net->layer_weights);
    }

    return net->layers[net->n_layers - 1]->nodes;
}

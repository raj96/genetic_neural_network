/**
 * @file
 * @brief Contains function which can perform operation on nn_net structure
 * @author Raja Nand Sharma
 */

#ifndef __NN_NET
#define __NN_NET

#include "nn/layer.h"

/**
 * @struct nn_net
 * @brief Defines a structure for the neural network
 */
typedef struct
{
    nn_layer **layers;     ///< Layers in the network
    int n_layers;          ///< Number of layers in the network
    float *layer_weights;  ///< Weights for all the layers in the network
    int n_layer_weights;   ///< Length of `layer_weights`
} nn_net;

/**
 * @brief Creates a nn_net structure
 */
nn_net *create_net();

/**
 * @brief Appends a layer to the nn_net struct passed
 * @param net Pointer to a nn_net structure, to which the layer is to be appended
 * @param[in] nnodes Number of nodes in the new layer
 * @param[in] a_fx Activation function for the layer
 */
void net_add_layer(nn_net *net, int nnodes, activation_fx a_fx);

/**
 * @brief Destroys the nn_net structure passed
 * @param[in] net Pointer to a nn_net structure to be destroyed
 */
void destroy_net(nn_net *net);

/**
 * @brief Performs a forward propagation on the whole network
 * @param net Pointer to a nn_net structure on which the forward propagation is to be performed
 */
float *forward_propagate_net(nn_net *net, float *input);

#endif
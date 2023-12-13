/**
 * @file
 * @brief Contains functions that can perform operations on nn_layer structs
 * @author Raja Nand Sharma
 */

#ifndef __NN_LAYER
#define __NN_LAYER

typedef float (*activation_fx)(float);

/**
 * @struct nn_layer
 * @brief	Defines structure of a layer in the neural network
 */
typedef struct {
    float *nodes;        ///< nodes in the layer
    activation_fx a_fx;  ///< activation function for the layer

    int w_i_start;  ///< weight index start for this layer, for the shared layer weights
    int _nweights;  ///< number of weights for this layer
    int _nnodes;    ///< number of nodes
} nn_layer;

/**
 * @brief Creates and returns a pointer to a layer struct
 * @param[in] n_nodes number of nodes in the current layer
 * @param[in] n_prev_nodes number of nodes in the previous layer; should be set to 0 for the first layer
 * @param[in] a_fx activation function for this layer
 * @param[in] layer_weights pointer to the shared layer weights structure
 * @param[in] n_layer_weights pointer to the length of layer_weights passed
 * @return pointer to a nn_layer struct
 */
nn_layer *create_layer(int n_nodes, int n_prev_nodes, activation_fx a_fx, float **layer_weights, int *n_layer_weights);

/**
 * @brief Destroys a layer
 * @param[in] layer pointer to the layer to destroy
 */
void destroy_layer(nn_layer *layer);

/**
 * @brief Performs a forward propagation from layer1 to layer2
 * @param layer1 pointer to a layer struct from which the values will be propagated
 * @param layer2 pointer to a layer struct to which the values will be propagated
 * @param layer_weights pointer to the shared layer weights structure
 */
void forward_propagate_layer(nn_layer *layer1, nn_layer *layer2, float *layer_weights);

#endif

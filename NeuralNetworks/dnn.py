"""
This program is trained using the Adam optmiser to minimise the loss function: 
    L = sum(pow(ddu + lambda*(u - pow(u,3))),2) / len(x) + c*pow(BC0,2) + c*pow(BC1,2)

The network is trained using the following functions: 
    plot_dnn: plots the DNN output as a function of the input
    model_grad: computes the gradient of the loss function with respect to the DNN parameters
    dnn: computes the DNN output given an input and the DNN parameters
    d_dnn: computes the derivative of the DNN output with respect to a specified component of the input
    dd_dnn: computes the second derivative of the DNN output with respect to a specified component of the input
    initialise_weights: initialises the DNN weights and biases
    initialise_he: initialises the DNN weights using the He initialisation method
    initialise_variables: initialises the iteration count, average gradient and average squared gradient

The network is trained using the following parameters:
    num_epoch: number of training epochs
    num_iter_per_epoch: number of iterations per epoch
    batch_size: batch size for training
    learnning_rate: learning rate for the optimizer
    layer_size: number of layers in the DNN
    neuron_num_per_layer: number of neurons in each layer except the output layer
    activation: activation function for the hidden layers except the output layer using the sine function, and it's first and second derivative. 
"""
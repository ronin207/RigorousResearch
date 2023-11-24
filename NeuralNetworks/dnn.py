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

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

class DeepNeuralNetwork:
    def __init__(self, num_epoch, num_iter_per_epoch, batch_size, learning_rate, layer_size, neuron_num_per_layer):
        self.num_epoch = num_epoch
        self.num_iter_per_epoch = num_iter_per_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.layer_size = layer_size
        self.neuron_num_per_layer = neuron_num_per_layer

    """
    DNN weights are initialised using the He initialisation method. 
    He initialisation method is known to be better suited for activation functions
    that are not symmetric around 0, such as the ReLU activation function. 
    The initialisation method sets the weight to be normally distributed with mean 0 and variance 2/n,
    where n is the number of inputs to the layer.

    Parameters:
        size (int): number of neurons in the layer
        num_inputs (int): number of inputs to the layer

    Returns:
        weights (tf.Variable): initialised weights
    """
    def initialise_he(self, size, num_inputs):
        weights = np.random.randn(size, num_inputs) * np.sqrt(2 / num_inputs)
        return tf.Variable(weights, dtype=tf.float32)
    
    def initialise_weights(self, n, layer_size):
        pass

    """
    Initialise the iteration count, average gradient and average squared gradient.

    Returns:
        iteration (int): iteration count
        ave_grad (list): average gradient
        ave_sqgrad (list): average squared gradient
    """
    def initialise_variables(self):
        iteration = 0
        ave_grad = []
        ave_sqgrad = []

        return iteration, ave_grad, ave_sqgrad 

    """
    Computes the forward propagation of the neural network. 
    The hidden layers are created using fully connected layers 
    with the specified number of neurons and activation function.
    The activation function utilises the sine function, and it's first and second derivative.

    Parameters:
        x (tf.Tensor): input
        p (list): DNN parameters
        act (list): activation function for each layer

    Returns:
        x (tf.Tensor): DNN output
    """
    @tf.function
    def dnn(self, x, p, act):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(self.neuron_num_per_layer, activation=act[0]['f'], input_shape=(1,)))

        for i in range(self.layer_size - 2):
            model.add(keras.layers.Dense(self.neuron_num_per_layer, activation=act[i+1]['f']))

        model.add(keras.layers.Dense(1, activation=act[len(p)-1]['f']))
        x = model(x)
        
        return x

    def d_dnn(self, x, p, act, component):
        pass

    def dd_dnn(self, x, p, act, component):
        pass

    @tf.function
    def model_grad(self, x, parameters, act):
        pass

    def plot_dnn(self, p, act):
        pass


if __name__ == "__main__":
    num_epochs = 50
    num_iter_per_epoch = 10
    batch_size = 100

    learning_rate = .01
    layer_size = 5
    neuron_num_per_layer = 50

    dnn = DeepNeuralNetwork(num_epochs, num_iter_per_epoch, batch_size, learning_rate, layer_size, neuron_num_per_layer)

    activation_fn = [{'f': tf.sin, 'df': tf.cos, 'ddf': lambda x: -tf.sin(x)} for _ in range(dnn.layer_size - 1)]
    parameters = dnn.initialise_weights(neuron_num_per_layer, layer_size)

    iteration, ave_grad, ave_sqgrad = dnn.initialise_variables()
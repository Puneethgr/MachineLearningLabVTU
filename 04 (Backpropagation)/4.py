# Program 4 (Backpropagation):

import numpy as np
from numpy.random import rand, randint 

inputNeurons = 2
hiddenLayerNeurons = 2
outputNeurons = 2

input_layer = randint(1,100,inputNeurons)

hidden_layer = rand(1,hiddenLayerNeurons)
hidden_bias = rand(1,hiddenLayerNeurons)
hidden_weights = rand(inputNeurons,hiddenLayerNeurons)

output = np.array([1,0])
output_bias = rand(1,outputNeurons)
output_weights = rand(hiddenLayerNeurons,outputNeurons)

def sigmoid(layer):
    return 1/(1 + np.exp(-layer))

def gradient(layer):
    return layer*(1-layer)

iterations = 2000
for i in range(iterations):
    hidden_layer = np.dot(input_layer,hidden_weights)
    hidden_layer = sigmoid(hidden_layer + hidden_bias)

    output_layer = np.dot(hidden_layer,output_weights)
    output_layer = sigmoid(output_layer + output_bias)

    error = output - output_layer

    error_terms_output = gradient(output_layer) * error
    error_terms_hidden = gradient(hidden_layer) * np.dot(error_terms_output, output_weights.T)

    gradient_output_weights = np.dot(hidden_layer.reshape(hiddenLayerNeurons,1), 
                                    error_terms_output.reshape(1,outputNeurons))
    gradient_hidden_weights = np.dot(input_layer.reshape(inputNeurons,1), 
                                    error_terms_hidden.reshape(1,hiddenLayerNeurons))

    output_weights += (0.05 * gradient_output_weights)  
    hidden_weights += (0.05 * gradient_hidden_weights)

    if i < 5 or i > iterations - 5:
        print("Iteration : ",i)
        print("Error : ",error)
        print("Output: ", output_layer) 
import numpy as np
import pandas as pd
#No machine learning libraries

#Loading the Training Data (5XOR split into 80/20 after randomisation and removal of headers)
training_data = pd.read_csv('C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/5XOR-Training.csv')
#print(training_data.head)
testing_data = pd.read_csv('C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/5XOR-Testing.csv')
#print(testing_data.head)

#Sigmoid function for the hidden and output layer neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Sigmoid derivative to find the gradient of the curve - to use in Error Backpropagation
#Backpropagation formula: error * sigmoid_derivative
def sigmoid_derivative(x):
    return x * (1 - x)

#Other Hyperparameters
learning_rate = 0.2
epochs = 10000
input_layer_size = training_data.shape[1]-1
hidden_layer_size = 4
output_layer_size = 1

#Training the Model
#Storing Each set of inputs in an array
inputs = training_data.iloc[:, :-1].values
#print(inputs) - 25rX5c matrix

#Storing Each inputs' expected output in an array
expected_outputs = training_data.iloc[:, -1].values
#print(expected_outputs) 

#Seeding numpy.random for reproducability
np.random.seed(10)
#Generating random weights for the inputs
hidden_weights = np.random.uniform(size=(input_layer_size,hidden_layer_size))
#Generating random weights for the hidden->output layer
output_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))

#Training algorithm 
for epoch in range(epochs):
    #Feed Forward Propagation
    #Input Later
    input_feedforward = np.dot(inputs,hidden_weights)
    hidden_layer_input = sigmoid(input_feedforward)
    #Hidden Layer
    hidden_feedforward = np.dot(hidden_layer_input,output_weights)
    output_layer_output = sigmoid(hidden_feedforward)
       
print(output_layer_output)
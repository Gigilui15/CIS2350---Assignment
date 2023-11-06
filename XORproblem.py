import numpy as np
import pandas as pd

#Loading the Training Data (5XOR split into 80/20 after randomisation and removal of headers)
training_data = pd.read_csv(
    'C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/5XOR-Training.csv'
    )

testing_data = pd.read_csv(
    'C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/5XOR-Testing.csv'
    )

#Sigmoid function for the hidden and output layer neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Sigmoid derivative to find the gradient of the curve - to use in Error Backpropagation
#Backpropagation formula: error * sigmoid_derivative
def sigmoid_derivative(x):
    return x * (1 - x)

#Other Hyperparameters
learning_rate = 0.2
epochs = 1000
input_layer_size = training_data.shape[1]-1
hidden_layer_size = 4
output_layer_size = 1

#Storing Each set of inputs in an array
inputs=training_data.iloc[:, :-1].values

#Storing Each inputs' expected output in an array
expected_outputs = training_data.iloc[:, -1].values

#Gathering the Testing Data
testing = testing_data.iloc[:, :-1].values
#Gathering the XOR column from the Testing dataset
testing_ans = testing_data.iloc[:, -1].values

#Seeding numpy.random for reproducability
np.random.seed(10)
#Generating random weights for the inputs
hidden_weights = np.random.uniform(size=(input_layer_size,hidden_layer_size))
#Generating random weights for the hidden->output layer
output_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))

error_threshold = 0.2
good_facts = []
bad_facts = []

#Training Algorithm 
for epoch in range(epochs):
    #Feed Forward Propagation
    #Input Layer
    input_feedforward = np.dot(inputs,hidden_weights)
    hidden_layer_input = sigmoid(input_feedforward)
    #Hidden Layer
    hidden_feedforward = np.dot(hidden_layer_input,output_weights)
    output_layer_output = sigmoid(hidden_feedforward)
    
    #Checking for the error
    error = expected_outputs.reshape(-1,1) - output_layer_output
    
    for i in range(len(error)):
        fact = {
            'Epoch': epoch + 1,
            'Index': i,
            'Expected Output': expected_outputs[i],
            'Output': output_layer_output[i][0],
            'Error': error[i][0]
        }
        
        if abs(error[i]) <= error_threshold:
            # Save good facts
            fact['Type'] = 'Good Fact'
            good_facts.append(fact)
        else:
            # Save bad facts
            fact['Type'] = 'Bad Fact'
            bad_facts.append(fact)

# Convert lists of dictionaries to Pandas DataFrames
good_df = pd.DataFrame(good_facts)
bad_df = pd.DataFrame(bad_facts)

# Save DataFrames as CSV files
good_df.to_csv('good_facts.csv', index=False)
bad_df.to_csv('bad_facts.csv', index=False)
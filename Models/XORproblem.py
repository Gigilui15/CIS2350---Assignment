import numpy as np
import pandas as pd
import pickle

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

def forward_propagation(sample, input_weights, hidden_weights):
    hidden_layer_input = np.dot(sample, input_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, hidden_weights)
    output_layer_output = sigmoid(output_layer_input)
    
    return hidden_layer_output, output_layer_output

def backpropagation(inputs, input_weights, hidden_weights, output_layer_output, outputs, hidden_layer_output):
    output_error = outputs - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = output_delta.dot(hidden_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    hidden_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
    input_weights += inputs.T.dot(hidden_delta) * learning_rate

    return hidden_weights, input_weights

# Function to calculate training accuracy
def training_accuracy(epoch, correct_predictions, total_examples):
    accuracy = (correct_predictions / total_examples) * 100
    print(f"\rTraining | Epoch: {epoch + 1} | Correct Predictions: {correct_predictions} | Accuracy: {accuracy:.2f}%", end="", flush=True)
    
#Hyperparameters
learning_rate = 0.2
epochs = 10000
input_layer_size = training_data.shape[1]-1
hidden_layer_size = 4
output_layer_size = 1
error_threshold = 0.2

#Storing Each set of inputs in an array
inputs = training_data.iloc[:, :-1].values
#Storing Each inputs' expected output in an array
expected_outputs = training_data.iloc[:, -1].values

#Gathering the Testing Data
testing = testing_data.iloc[:, :-1].values
#Gathering the XOR column from the Testing dataset
testing_ans = testing_data.iloc[:, -1].values

#Seeding numpy.random for reproducability
np.random.seed(666)
#Generating random weights for the inputs
input_weights = np.random.uniform(size=(input_layer_size,hidden_layer_size))

#Generating random weights for the hidden->output layer
hidden_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))

facts = []

# Training Algorithm
converged = False
convergence_epoch = None
for epoch in range(epochs):
    bad_facts_count = 0
    correct_predictions = 0

    for input_index in range(len(inputs)):
        sample = inputs[input_index].reshape(1, -1)
        hidden_layer_output, output_layer_output = forward_propagation(sample, input_weights, hidden_weights)
        error = expected_outputs[input_index] - output_layer_output

        if abs(error) > error_threshold:
            backpropagation(sample, input_weights, hidden_weights, output_layer_output, expected_outputs[input_index], hidden_layer_output)
            bad_facts_count += 1
            status = "Bad Fact"
        else:
            status = "Good Fact"
            correct_predictions += 1

        fact = {
            'Epoch': epoch + 1,
            'Index': input_index + 1,
            'Answer': expected_outputs[input_index],
            'Output': output_layer_output[0],
            'Error': error[0],
            'Fact': status
        }
        facts.append(fact)

   # Call the training accuracy function
    training_accuracy(epoch, correct_predictions, len(inputs))

    # Check for convergence by examining bad facts in the last epoch
    if bad_facts_count == 0:
        converged = True
        convergence_epoch = epoch + 1
        break  # Break out of the training loop

# Print convergence information
if converged:
    print(f"\nConvergence reached at epoch {convergence_epoch} based on training accuracy.")
else:
    print("\nTraining completed without convergence.")

# Testing Algorithm
correct_predictions = 0
print("\nTraining:\n")
for input_index in range(len(testing)):
    test_sample = testing[input_index].reshape(1, -1)
    test_hidden_layer_output, test_output_layer_output = forward_propagation(test_sample, input_weights, hidden_weights)
    test_error = testing_ans[input_index] - test_output_layer_output
    if abs(test_error) > error_threshold:
        test_status = "Incorrect"
    else:
        test_status = "Correct"
        correct_predictions += 1
        
    test_fact = {
        'Index': input_index + 1,
        'Answer': testing_ans[input_index],
        'Output': test_output_layer_output[0][0],  
        'Fact': test_status
    }

    # Print the test fact to the console without curly braces and single quotes
    print(f"Test Fact {test_fact['Index']} | Answer = {test_fact['Answer']} | Output = {test_fact['Output']:.6f} | {test_fact['Fact']}")

# Calculate and print the percentage accuracy
accuracy = (correct_predictions / len(testing)) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

print("\n Saving Facts and Weights...")
# Convert the list of facts to a DataFrame
facts_df = pd.DataFrame(facts)

# Save the DataFrame to a CSV file
facts_df.to_csv('training_facts.csv', index=False)

# Saving the Weights
with open('input_weights.pkl', 'wb') as f:
    pickle.dump(input_weights, f)

with open('hidden_weights.pkl', 'wb') as f:
    pickle.dump(hidden_weights, f)
    
print("\n Finished!")
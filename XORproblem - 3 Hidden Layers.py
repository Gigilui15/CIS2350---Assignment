import numpy as np
import pandas as pd
import pickle

# Loading the Training Data (5XOR split into 80/20 after randomization and removal of headers)
training_data = pd.read_csv('C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/5XOR-Training.csv')

testing_data = pd.read_csv('C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/5XOR-Testing.csv')

# Sigmoid function for the hidden and output layer neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative to find the gradient of the curve - to use in Error Backpropagation
# Backpropagation formula: error * sigmoid_derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Other Hyperparameters
learning_rate = 0.2
epochs = 30000
input_layer_size = training_data.shape[1] - 1
hidden_layer1_size = 2
hidden_layer2_size =  3# Second hidden layer
hidden_layer3_size = 4  # New hidden layer
output_layer_size = 1

# Storing Each set of inputs in an array
inputs = training_data.iloc[:, :-1].values

# Storing Each input's expected output in an array
expected_outputs = training_data.iloc[:, -1].values

# Gathering the Testing Data
testing = testing_data.iloc[:, :-1].values
# Gathering the XOR column from the Testing dataset
testing_ans = testing_data.iloc[:, -1].values

# Seeding numpy.random for reproducibility
np.random.seed(200)

# Generating random weights for the inputs to the first hidden layer
hidden_weights1 = np.random.uniform(size=(input_layer_size, hidden_layer1_size))

# Generating random weights for the connections between the first and second hidden layers
hidden_weights2 = np.random.uniform(size=(hidden_layer1_size, hidden_layer2_size))

# Generating random weights for the connections between the second and third hidden layers
hidden_weights3 = np.random.uniform(size=(hidden_layer2_size, hidden_layer3_size))

# Generating random weights for the third hidden layer to the output layer
output_weights = np.random.uniform(size=(hidden_layer3_size, output_layer_size))

error_threshold = 0.2
good_facts = []
bad_facts = []

# Training Algorithm
for epoch in range(epochs):
    # Feed Forward Propagation
    # Input Layer to First Hidden Layer
    input_feedforward1 = np.dot(inputs, hidden_weights1)
    hidden_layer1_input = sigmoid(input_feedforward1)

    # First Hidden Layer to Second Hidden Layer
    input_feedforward2 = np.dot(hidden_layer1_input, hidden_weights2)
    hidden_layer2_input = sigmoid(input_feedforward2)

    # Second Hidden Layer to Third Hidden Layer
    input_feedforward3 = np.dot(hidden_layer2_input, hidden_weights3)
    hidden_layer3_input = sigmoid(input_feedforward3)

    # Third Hidden Layer to Output Layer
    hidden_feedforward = np.dot(hidden_layer3_input, output_weights)
    output_layer_output = sigmoid(hidden_feedforward)

    # Checking for the error
    error = expected_outputs.reshape(-1, 1) - output_layer_output

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

            # Perform backpropagation only when there is a "bad fact"
            delta_output = error[i] * sigmoid_derivative(output_layer_output[i])
            error_hidden3 = delta_output.dot(output_weights.T)
            delta_hidden3 = error_hidden3 * sigmoid_derivative(hidden_layer3_input[i])

            error_hidden2 = delta_hidden3.dot(hidden_weights3.T)
            delta_hidden2 = error_hidden2 * sigmoid_derivative(hidden_layer2_input[i])

            error_hidden1 = delta_hidden2.dot(hidden_weights2.T)
            delta_hidden1 = error_hidden1 * sigmoid_derivative(hidden_layer1_input[i])

            output_weights += hidden_layer3_input[i].reshape(-1, 1).dot(delta_output.reshape(1, -1)) * learning_rate
            hidden_weights3 += hidden_layer2_input[i].reshape(-1, 1).dot(delta_hidden3.reshape(1, -1)) * learning_rate
            hidden_weights2 += hidden_layer1_input[i].reshape(-1, 1).dot(delta_hidden2.reshape(1, -1)) * learning_rate
            hidden_weights1 += inputs[i].reshape(-1, 1).dot(delta_hidden1.reshape(1, -1)) * learning_rate

# Save the trained weights in a pickle file
with open('model_weights.pkl', 'wb') as file:
    pickle.dump((hidden_weights1, hidden_weights2, hidden_weights3, output_weights), file)

# Convert lists of dictionaries to Pandas DataFrames
good_df = pd.DataFrame(good_facts)
bad_df = pd.DataFrame(bad_facts)

# Save DataFrames as CSV files
good_df.to_csv('good_facts.csv', index=False)
bad_df.to_csv('bad_facts.csv', index=False)

# Function to make predictions using the trained neural network
def predict(input_data, hidden_weights1, hidden_weights2, hidden_weights3, output_weights):
    hidden_layer1_input = sigmoid(np.dot(input_data, hidden_weights1))
    hidden_layer2_input = sigmoid(np.dot(hidden_layer1_input, hidden_weights2))
    hidden_layer3_input = sigmoid(np.dot(hidden_layer2_input, hidden_weights3))
    output_layer_output = sigmoid(np.dot(hidden_layer3_input, output_weights))
    return output_layer_output

# Perform predictions on the testing_data
testing_predictions = predict(testing, hidden_weights1, hidden_weights2, hidden_weights3, output_weights)

# Compare the predictions with the actual testing data
for i in range(len(testing_predictions)):
    print(f"Test {i + 1} - Predicted: {testing_predictions[i][0]}, Actual: {testing_ans[i]}")

# Function to compute the accuracy based on the specified criteria
def compute_accuracy(predictions, actual):
    correct_predictions = 0

    for i in range(len(predictions)):
        if (predictions[i][0] < 0.5 and actual[i] == 0) or (predictions[i][0] >= 0.5 and actual[i] == 1):
            correct_predictions += 1

    accuracy = correct_predictions / len(predictions)
    return accuracy
# Compute and print the accuracy
accuracy = compute_accuracy(testing_predictions, testing_ans)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
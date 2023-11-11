import pandas as pd

# Read your CSV file into a pandas DataFrame
# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/CIS2350---Assignment/bad_facts.csv')

# Calculate the count of unique "Epoch" IDs
epoch_counts = df['Epoch'].value_counts().reset_index()
epoch_counts.columns = ['Epoch ID', 'Count']

# Save the result as a CSV file
# Replace 'epoch_counts.csv' with the desired output file name
epoch_counts.to_csv('epoch_counts.csv', index=False)

print("Epoch counts saved as 'epoch_counts.csv'")

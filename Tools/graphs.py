import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/CIS2350---Assignment/training_facts.csv'

# Read the Excel sheet into a DataFrame
df = pd.read_csv(excel_file_path)

# Filter the DataFrame to include only 'Epoch' and 'Fact' columns
filtered_df = df[['Epoch', 'Fact']]

# Count the number of 'Bad Fact' entries for each epoch
bad_facts_count = filtered_df[filtered_df['Fact'] == 'Bad Fact'].groupby('Epoch').size().reset_index(name='No of Bad Facts')

# Specify the path to save the results Excel file
output_excel_path = 'C:/Users/luigi/Desktop/Third Year/Business Applications of AI/Assignment/CIS2350---Assignment/bad_facts_results.xlsx'

# Save the results to a new Excel file
bad_facts_count.to_excel(output_excel_path, index=False)

print(f"Results saved to {output_excel_path}")

# Note: successful operation of importing CSV files into Python!!!!

import pandas as pd

# Importing a CSV file
data = pd.read_csv('/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Modules/Week 7/Sprint 2/Datasets/dataset.csv')


print(data)

# Display the first few rows
print(data.head())

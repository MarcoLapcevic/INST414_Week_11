# Note: This is a simple example of a Decision Tree model using a dataset from Kaggle.

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load the dataset (Ensure the path is correct)
data = pd.read_csv('/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Modules/Week 7/Sprint 2/Datasets/Kaggle Datasets/dataset.csv')

# Print column names to verify
print("Dataset Columns:", data.columns)

# Ensure selected features exist and drop missing values first
data = data.dropna(subset=['Curricular units 2nd sem (approved)', 'Age at enrollment', 'Target'])

# Define independent and dependent variables
x = data[['Curricular units 2nd sem (approved)', 'Age at enrollment']]
y = data['Target'].astype(str)  # Convert to string for classification

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Ensure correct class names for visualization
class_names = [str(c) for c in y.unique()]

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['Curricular units 2nd sem (approved)', 'Age at enrollment'], class_names=class_names, filled=True)
plt.title('Simple Decision Tree Visualization')
plt.show()

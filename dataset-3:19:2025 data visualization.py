# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Modules/Week 7/Sprint 2/Datasets/dataset.csv')

# Select two quantitative variables (assuming "Variable1" and "Variable2" are quantitative)
x = data[["Mother's qualification"]]  # Independent variable
y = data['Age at enrollment']    # Dependent variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict values
y_pred = model.predict(x)

# Visualize the linear regression with a scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data["Mother's qualification"], y=data['Age at enrollment'], label='Actual Data')
plt.plot(data["Mother's qualification"], y_pred, color='red', label='Regression Line')
plt.title("Linear Regression between Mother's qualification and Age at enrollment")
plt.xlabel("Mother's qualification")
plt.ylabel('Age at enrollment')
plt.legend()
plt.show()

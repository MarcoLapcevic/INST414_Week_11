# Student Performance Prediction using Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Step 1: Data Gathering and Loading
file_path = '/mnt/data/dataset.csv'
data = pd.read_csv(file_path)

# Step 2: Data Dictionary Creation
print('Data Dictionary:')
data_dictionary = pd.DataFrame({
    'Variable': data.columns,
    'Data Type': data.dtypes.values,
    'Description': ['N/A'] * len(data.columns)
})
print(data_dictionary)

# Step 3: Data Cleaning and Preprocessing
print('\nData Cleaning and Preprocessing...')
missing_values = data.isnull().sum()
unique_values = data.nunique()
data_overview = pd.DataFrame({
    'Missing Values': missing_values,
    'Unique Values': unique_values,
    'Data Type': data.dtypes
})
print(data_overview)

# Step 4: Encoding Categorical Variables
label_encoder = LabelEncoder()
categorical_columns = [
    'Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'International'
]
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
data['Target'] = label_encoder.fit_transform(data['Target'])

# Step 5: Normalization
scaler = StandardScaler()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 6: Data Visualization - Correlation Heatmap
plt.figure(figsize=(15, 12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Step 7: Feature Importance using Random Forest
X = data.drop('Target', axis=1)
y = data['Target']
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot Feature Importances
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[:15], y=feature_importances.index[:15], palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Step 8: Baseline Model Implementation and Evaluation
print('\nBaseline Model Implementation...')
top_features = feature_importances.index[:10]
X_top = data[top_features]
cv_scores = cross_val_score(rf, X_top, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean():.4f}')
print(f'Standard Deviation: {cv_scores.std():.4f}')

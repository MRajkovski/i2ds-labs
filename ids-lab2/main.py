import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import missingno as msno


# Step 1
# Read dataset
df = pd.read_csv('diabetes.csv')

# Step 2
# Check for missing values
missing_values = df.isnull().sum()

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(df)) * 100

missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

#print(missing_data)

# Step 3
# Visualize missing values with a matrix
plt.figure(figsize=(12, 6))
msno.matrix(df)
#plt.show()

# Visualize missing values with a heatmap
plt.figure(figsize=(12, 6))
msno.heatmap(df)
#plt.show()

# Visualize the missing values dendrogram (showing the hierarchical clustering of missing data)
plt.figure(figsize=(12, 6))
msno.dendrogram(df)
#plt.show()

# Step 4
# Handle missing values
df.fillna(df.mean(), inplace=True)
#print(df.isnull().sum())

# Step 5
# Print the first rows of your final Dataset
df.to_csv('diabetes_cleaned.csv', index=False)
print(df.head())

# Step 6
# Assuming the target column is 'Outcome' and the features are all other columns
X = df.drop(columns=['Outcome'])  # Features (all columns except 'Outcome')
y = df['Outcome']  # Target column ('Outcome')

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, check the shape of the split datasets
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Step 7
# Initialize the KNN Classifier with n_neighbors=5 (you can change the number of neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
# Train the KNN model using the training data (X_train, y_train)
knn.fit(X_train, y_train)

# Step 8
# Predict the outcomes on the test set
y_pred = knn.predict(X_test)

# Step 9
# Generate the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
# Generate the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Calculate accuracy on the test set
print("Accuracy Score:", accuracy_score(y_test, y_pred))
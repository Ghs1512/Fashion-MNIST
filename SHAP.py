import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import shap

# Load the data with memory mapping to avoid memory issues
X = np.load("C:/Users/prajw/AI_features_pls_work.npy", mmap_mode='r')
y = np.load("C:/Users/prajw/AI_labels_pls_work.npy", mmap_mode='r')

# Use a subset of the data
num_samples = 60000  # Adjust based on memory capacity
X_subset = X[:num_samples]
y_subset = y[:num_samples]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.5, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='linear', probability=True)  # Enable probability for SHAP
svm.fit(X_train, y_train)

# Predict labels for the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create SHAP explainer
explainer = shap.KernelExplainer(svm.predict_proba, X_train[:100])  # Use a smaller subset of training data for efficiency
shap_values = explainer.shap_values(X_test[:100])  # Use a subset of test data to calculate SHAP values

# Plot summary plot of SHAP values
shap.summary_plot(shap_values, X_test[:100], feature_names=[f"Feature {i}" for i in range(X_test.shape[1])])

# Calculate the mean absolute SHAP values for each feature
shap_mean = np.abs(shap_values).mean(axis=1).mean(axis=0)

# Create a DataFrame for feature importance
shap_df = pd.DataFrame({
    "Feature": [f"Feature {i}" for i in range(X_test.shape[1])],
    "Mean SHAP Value": shap_mean
})

# Get the top 10 most important features
top_features = shap_df.nlargest(10, "Mean SHAP Value")

# Plot a bar chart for the top features
plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, y="Feature", x="Mean SHAP Value", palette="viridis")
plt.title("Top 10 Features by Mean SHAP Value", fontsize=16)
plt.xlabel("Mean SHAP Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# Load the features and labels
X = np.load("D:\\ASEB\\AI\\Iv3_features_pls_work.npy", mmap_mode='r')  # Use memory mapping
y = np.load("D:\\ASEB\\AI\\Iv3_labels_pls_work.npy", mmap_mode='r')  # Use memory mapping

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define batch size to control memory usage
batch_size = 5000  # Adjust based on available memory
num_samples = X_train.shape[0]

# Initialize Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)

# Prepare batches
print("Training Decision Tree in batches...")
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]

    # Fit the Decision Tree on each batch
    # Since DecisionTreeClassifier does not support incremental training,
    # we refit the model for every batch
    dt.fit(X_batch, y_batch)

# Predict and evaluate the model on the test set
y_pred_dt = dt.predict(X_test)

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

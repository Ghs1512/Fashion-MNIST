import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier  # Supports incremental learning

# Load the features and labels
X = np.load("D:\\ASEB\\AI\\Iv3_features_pls_work.npy", mmap_mode='r')  # Use memory mapping
y = np.load("D:\\ASEB\\AI\\Iv3_labels_pls_work.npy", mmap_mode='r')  # Use memory mapping

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define batch size to control memory usage
batch_size = 1000
num_samples = X_train.shape[0]

# Initialize an SGDClassifier as an incremental version of linear SVM
svm = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=42)  # Linear kernel

# Train the model incrementally with batch processing
print("Training SVM in batches...")
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]
    svm.partial_fit(X_batch, y_batch, classes=np.unique(y))  # Use partial_fit for incremental learning

# Predict and evaluate the model on the test set
y_pred_svm = svm.predict(X_test)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

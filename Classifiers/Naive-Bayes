import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

# Load the features and labels
X = np.load("D:\\ASEB\\AI\\Iv3_features_pls_work.npy", mmap_mode='r')  # Use memory mapping
y = np.load("D:\\ASEB\\AI\\Iv3_labels_pls_work.npy", mmap_mode='r')  # Use memory mapping

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define batch size to control memory usage
batch_size = 1000
num_samples = X_train.shape[0]

# Initialize Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model incrementally with batch processing
print("Training Naive Bayes in batches...")
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]
    # Fit the model incrementally using partial batches
    if start == 0:
        # For the first batch, fit the model
        nb_model.fit(X_batch, y_batch)
    else:
        # For subsequent batches, update the model incrementally
        nb_model.partial_fit(X_batch, y_batch, classes=np.unique(y))

# Predict and evaluate the model on the test set
y_pred_nb = nb_model.predict(X_test)

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Load the features and labels
X = np.load("D:\\ASEB\\AI\\Iv3_features_pls_work.npy", mmap_mode='r')  # Use memory mapping
y = np.load("D:\\ASEB\\AI\\Iv3_labels_pls_work.npy", mmap_mode='r')  # Use memory mapping

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define batch size to control memory usage
batch_size = 5000  # Adjust based on available memory
num_samples = X_train.shape[0]

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Adjust n_estimators as needed

# Train the model incrementally with batch processing
print("Training Random Forest in batches...")
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]

    # Train the Random Forest on each batch
    if start == 0:
        # Fit the model on the first batch
        rf.fit(X_batch, y_batch)
    else:
        # Combine the previous model with the new batch
        rf.n_estimators += 100  # Add more trees to the existing model
        rf.fit(X_batch, y_batch)  # Retrain with new data

# Predict and evaluate the model on the test set
y_pred_rf = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


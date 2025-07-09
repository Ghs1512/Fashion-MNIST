import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import classifiers
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the features and labels
X = np.load("D:\\ASEB\\AI\\Iv3_features_pls_work.npy")
y = np.load("D:\\ASEB\\AI\\Iv3_labels_pls_work.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Classification report for KNN
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

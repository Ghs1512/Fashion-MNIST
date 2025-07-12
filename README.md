# Fashion-MNIST

The proposed work in this study is a deep learning model for
the classification of ten categories of fashion items. The model
attains a maximum accuracy of 0.98 with a Support Vector
Machine (SVM) and Inception v3 architecture. To improve model
interpretability, Explainable AI (XAI) methods like SHapley
Additive exPlanations (SHAP) and Local Interpretable Model-
agnostic Explanations (LIME) are utilized. LIME heatmaps
reveal important image areas that impact the model’s predictions,
whereas SHAP summary plots indicate the most important
features that drive classification decisions.

<img width="448" alt="image" src="https://github.com/user-attachments/assets/fa08a2e8-0f8a-4eff-a77d-e837faf93a8a" />


📊 Performance Comparison of CNN Models with Different Classifiers
Table 1: CNN + KNN, SVM, Naïve Bayes
CNN \ Classifier	KNN Accuracy	KNN Precision	KNN Recall	KNN F1-Score	SVM Accuracy	SVM Precision	SVM Recall	SVM F1-Score	Naïve Bayes Accuracy	Naïve Bayes Precision	Naïve Bayes Recall	Naïve Bayes F1-Score
VGG16	0.87	0.87	0.87	0.87	0.89	0.89	0.89	0.89	0.77	0.78	0.77	0.76
Inception V3	0.98	0.98	0.98	0.98	0.98	0.98	0.98	0.98	0.96	0.95	0.96	0.95
MobileNet V2	0.87	0.87	0.87	0.87	0.85	0.87	0.85	0.85	0.78	0.78	0.78	0.77
ResNet50	0.97	0.96	0.96	0.96	0.96	0.95	0.96	0.96	0.89	0.88	0.85	0.86
EfficientNet B0	0.87	0.87	0.87	0.85	0.87	0.85	0.85	0.85	0.76	0.76	0.76	0.75
Table 2: CNN + Random Forest, Decision Tree, XGBoost
CNN \ Classifier	Random Forest Accuracy	Random Forest Precision	Random Forest Recall	Random Forest F1-Score	Decision Tree Accuracy	Decision Tree Precision	Decision Tree Recall	Decision Tree F1-Score	XGBoost Accuracy	XGBoost Precision	XGBoost Recall	XGBoost F1-Score
VGG16	0.86	0.85	0.85	0.86	0.76	0.76	0.76	0.76	0.91	0.90	0.91	0.90
Inception V3	0.97	0.97	0.97	0.97	0.94	0.93	0.94	0.93	0.95	0.93	0.95	0.93
MobileNet V2	0.85	0.85	0.84	0.85	0.75	0.75	0.75	0.75	0.83	0.83	0.81	0.81
ResNet50	0.96	0.94	0.95	0.94	0.93	0.92	0.93	0.92	0.94	0.94	0.95	0.94
EfficientNet B0	0.83	0.83	0.83	0.83	0.70	0.70	0.70	0.70	0.81	0.80	0.80

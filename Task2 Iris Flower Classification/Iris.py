# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()
data = pd.read_csv('/content/drive/MyDrive/lris/IRIS.csv')


# Convert to DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target
data['Species'] = data['Species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Display first rows
print(data.head())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Encode target labels
encoder = LabelEncoder()
data['Species_encoded'] = encoder.fit_transform(data['Species'])

# Features & Target
X = data[['sepal length (cm)', 'sepal width (cm)',
          'petal length (cm)', 'petal width (cm)']]
y = data['Species_encoded']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Visualization
# -----------------------------

# Confusion Matrix Heatmap
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Classification")
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
plt.figure()
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")
plt.show()

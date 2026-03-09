model_logistic = LogisticRegression()
model_logistic.fit(X_train_logistic, Y_train_logistic)

Y_pred_logistic = model_logistic.predict(X_test_logistic)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

accuracy = accuracy_score(Y_test_logistic, Y_pred_logistic)
precision = precision_score(Y_test_logistic, Y_pred_logistic)
recall = recall_score(Y_test_logistic, Y_pred_logistic)
f1 = f1_score(Y_test_logistic, Y_pred_logistic)

print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(Y_test_logistic, Y_pred_logistic))

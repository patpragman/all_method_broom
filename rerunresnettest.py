from sklearn.metrics import classification_report

# reconstruct classification report from confusion matrix because I forgot to do it
y_true = [1 for _ in range(0, 90)] + [1 for _ in range(0, 253)] + [0 for _ in range(0, 560)] + [0 for _ in range(0, 65)]
y_pred = [0 for _ in range(0, 90)] + [1 for _ in range(0, 253)] + [0 for _ in range(0, 560)] + [1 for _ in range(0, 65)]

cr = classification_report(y_true, y_pred)
print(cr)
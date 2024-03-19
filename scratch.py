# True Positives, False Positives, False Negatives, True Negatives
TP = 273
FP = 77
FN = 73
TN = 545

# Calculate precision, recall and F1 score for Not Entangled (Class 0)
precision_not_entangled = TN / (TN + FN)
recall_not_entangled = TN / (TN + FP)
f1_not_entangled = 2 * (precision_not_entangled * recall_not_entangled) / (precision_not_entangled + recall_not_entangled)

# Calculate precision, recall and F1 score for Entangled (Class 1)
precision_entangled = TP / (TP + FP)
recall_entangled = TP / (TP + FN)
f1_entangled = 2 * (precision_entangled * recall_entangled) / (precision_entangled + recall_entangled)

# Calculate Macro F1 Score
macro_f1 = (f1_not_entangled + f1_entangled) / 2

print(precision_not_entangled, recall_not_entangled, f1_not_entangled, precision_entangled, recall_entangled, f1_entangled, macro_f1)

print(TP/ (TP + FN))
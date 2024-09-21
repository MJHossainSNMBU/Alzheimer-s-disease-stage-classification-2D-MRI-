
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions on the test set
BaseNetPrediction = model.predict_generator(test_generator)  # Predictions on test data
BaseNetPrediction = np.argmax(BaseNetPrediction, axis=1)

# Calculate accuracy
accuracyBaseNet = accuracy_score(test_generator.classes, BaseNetPrediction)
print(f'AlzheimerNet model test accuracy is: {(accuracyBaseNet*100):.2f}%')

# Creating confusion matrix
batch_size = 16
test_samples = 1200
class_Names = test_generator.class_indices
Y_pred_baseNet = model.predict_generator(test_generator, test_samples // batch_size + 1)
y_pred_baseNet = np.argmax(Y_pred_baseNet, axis=1)

class_Names = {v: k for k, v in class_Names.items()}
class_names_CNN = ['Mild_Demented', 'Non_Demented', 'Very_Mild_Demented']

# Function to plot confusion matrix heatmap
def plot_heatmap(y_true, y_pred, class_names, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, square=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap=plt.cm.Blues, cbar=False, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

# Plotting the confusion matrix
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
tClasses = test_generator.classes
plot_heatmap(tClasses, y_pred_baseNet, class_names_CNN, ax1, title="AlzheimerNet")
fig.tight_layout()
fig.subplots_adjust(top=1.25)
plt.show()

# Generating classification report
class_labels = test_generator.class_indices
class_labels_baseNet = {v: k for k, v in class_labels.items()}
target_names = ['Mild_Demented', 'Non_Demented', 'Very_Mild_Demented']
print('\nClassification report: Pretrained BaseNet\n')
print(classification_report(test_generator.classes, y_pred_baseNet, target_names=target_names))

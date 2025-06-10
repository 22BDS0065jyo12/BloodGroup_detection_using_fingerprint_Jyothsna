import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = load_model(r"C:\Users\dell\Downloads\fingerprint_blood_group_detection\blood_group_cnn_model.h5")

# Data generator for validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    directory=r"C:\Users\dell\Downloads\fingerprint_blood_group\fingerprint_blood_group\dataset",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Make predictions
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes 

class_labels = list(val_generator.class_indices.keys())

# Print metrics
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

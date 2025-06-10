# 🧬 Fingerprint-Based Blood Group Detection using CNN

This project implements a Convolutional Neural Network (CNN) model using TensorFlow and OpenCV to detect human blood groups from fingerprint images. It includes image preprocessing, model training, evaluation, and saving the final model.

## 📁 Dataset Folder Structure

Each folder in the dataset corresponds to a blood group label.

| Blood Group | Folder Name |
|-------------|-------------|
| A Positive  | `A+/`       |
| A Negative  | `A-/`       |
| B Positive  | `B+/`       |
| B Negative  | `B-/`       |
| AB Positive | `AB+/`      |
| AB Negative | `AB-/`      |
| O Positive  | `O+/`       |
| O Negative  | `O-/`       |


## 📸 Sample Dataset Screenshot

![Dataset Screenshot](screenshots/A+.png)

## 📌 Project Workflow

### 1. **Image Preprocessing**

- Read and resize each image to `128x128`.
- Normalize pixel values to the range `[0, 1]`.
- Assign a label to each image based on folder name.
- Convert labels to one-hot encoded vectors.
- Save the processed data as `.npy` files.

**Screenshot:**
![Preprocessing Screenshot](screenshots/O-.png)
![A-](screenshots/A-.png)
![O+ Screenshot](screenshots/O+.png)
![B- Screenshot](screenshots/B-.png)
![B+ Screenshot](screenshots/B+.png)
### 2. **Model Architecture**

A simple CNN model with the following layers:

- **Conv2D** → ReLU
- **MaxPooling2D**
- **Conv2D** → ReLU
- **MaxPooling2D**
- **Flatten**
- **Dense** → ReLU
- **Dropout**
- **Dense (Softmax)**

**Screenshot:**
![Model Summary Screenshot](screenshots/intital_page.png)
![Confsusion Matrix](screenshots/Screenshot 2025-06-10 115517.png)

### 3. **Model Training**

- Model is trained using 80% training and 20% validation data.
- Loss function: `categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 10
- Batch Size: 32

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

4. Model Evaluation
Loads the trained .h5 model.
Evaluates using a validation split via ImageDataGenerator.

5. Model Saving
Final model is saved as: blood_group_cnn_model.h5
Use load_model() from Keras to reload the trained model.

⚙️ Requirements
Install the required libraries with: pip install tensorflow opencv-python numpy scikit-learn

## 📁 File Structure

| File/Folder             | Description                                                |
|-------------------------|------------------------------------------------------------|
| `dataset/`              | Contains folders for each blood group (A+, A-, B+, etc.)   |
| `screenshots/`          | Contains all screenshots used in the README                |
| ├── `dataset_structure.png`     | Visual of dataset directory tree               |
| ├── `preprocessing_output.png`  | Output after preprocessing sample image        |
| ├── `model_summary.png`         | Summary of the CNN model architecture          |
| ├── `training_accuracy.png`     | Accuracy/loss graph during training             |
| └── `confusion_matrix.png`      | Confusion matrix after evaluation               |
| `preprocess.py`         | Script to preprocess images and save them as `.npy`        |
| `train_model.py`        | Script to build, train, and save the CNN model             |
| `evaluate_model.py`     | Script to load the model and evaluate on validation data   |
| `X_train.npy`           | Numpy array of preprocessed training images                |
| `X_val.npy`             | Numpy array of validation images                           |
| `y_train.npy`           | One-hot labels for training data                           |
| `y_val.npy`             | One-hot labels for validation data                         |
| `blood_group_cnn_model.h5` | Saved trained Keras model                         |
| `README.md`             | This file – contains project overview and instructions      |

📊 Results
Trained with 10 epochs

Achieved decent accuracy on validation set of upto 88%


from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = load_model("blood_group_cnn_model.h5")

# Blood group class labels
class_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Upload folder path
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Sanitize the filename
    filename = secure_filename(file.filename)

    # Full path to save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the uploaded file
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize as done during training

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return render_template('result.html', prediction=predicted_class, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

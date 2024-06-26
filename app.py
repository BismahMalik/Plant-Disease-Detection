from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load the trained model
model = load_model('plant_disease_prediction_model.h5')

# Load class indices
class_indices = json.load(open('class_indices.json'))

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        # Preprocess the image
        processed_image = preprocess_image(file)
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_indices[str(predicted_class_index)]
        # Return prediction result
        return jsonify({'class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import base64
from PIL import Image
import io

app = Flask(__name__)

class FashionMNISTPredictor:
    def __init__(self, model_path='fashion_mnist_model.h5', class_names_path='class_names.json'):
        """Initialize the predictor with trained model and class names"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Model loaded successfully from {model_path}")
            print(f"Classes: {self.class_names}")
        except Exception as e:
            raise Exception(f"Failed to load model or class names: {e}")
    
    def preprocess_image(self, image_data):
        """Preprocess base64 image data for prediction"""
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict(self, image_data, top_k=5):
        """Make prediction on base64 image data"""
        # Preprocess image
        img_array = self.preprocess_image(image_data)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx]
            confidence = float(probabilities[idx] * 100)
            results.append({
                'class': class_name,
                'confidence': confidence,
                'index': int(idx)
            })
        
        return results

# Initialize predictor
try:
    predictor = FashionMNISTPredictor()
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        image_data = data['image']
        
        # Make prediction
        results = predictor.predict(image_data)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
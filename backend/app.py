"""
Flask backend API for PA228 image segmentation project
Provides endpoints for model inference and image processing
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import plot_pred
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
CORS(app)

# Global variable to store the model
model = None
device = torch.device('cpu')

def load_model(model_path='model.pt'):
    """Load the trained model"""
    global model, device
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
        model.eval()
        return True
    return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Perform inference on uploaded image
    Expects: multipart/form-data with 'image' file
    Returns: JSON with base64 encoded prediction image
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)
        
        # Apply transformations
        transforms = A.Compose([
            A.SmallestMaxSize(512),
            A.CenterCrop(512, 1024),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        transformed = transforms(image=img_np)
        img_tensor = transformed['image'].unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            pred = model(img_tensor)
        
        # Convert prediction to RGB image
        rgb_tensor = plot_pred(pred)
        rgb_np = rgb_tensor.numpy()
        
        # Convert to base64
        pred_img = Image.fromarray(rgb_np)
        buffered = io.BytesIO()
        pred_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': img_str,
            'shape': list(rgb_np.shape)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    return jsonify({
        'loaded': True,
        'device': str(device),
        'classes': 8,
        'class_labels': [
            'unlabeled',
            'road',
            'building',
            'wall',
            'vegetation',
            'sky',
            'vehicle',
            'traffic'
        ]
    })

if __name__ == '__main__':
    # Try to load model on startup
    model_path = os.environ.get('MODEL_PATH', 'model.pt')
    if load_model(model_path):
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Server will start but predictions will fail until model is loaded")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

# PA228 Backend API

Flask-based REST API for serving PyTorch image segmentation models.

## Features

- RESTful API for model inference
- CORS enabled for frontend communication
- Automatic model loading
- Image preprocessing and postprocessing
- Base64 image encoding for easy web transport

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Configuration

The backend can be configured using environment variables:

- `MODEL_PATH`: Path to the trained PyTorch model file (default: `model.pt` in project root)
- `PORT`: Port to run the server on (default: `5000`)

Example:
```bash
export MODEL_PATH=/path/to/your/model.pt
export PORT=8000
```

## Running the Server

### Development Mode

```bash
python app.py
```

The server will start on `http://localhost:5000` with debug mode enabled.

### Production Mode

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Health Check

Check if the server and model are running.

```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Model Information

Get information about the loaded model.

```
GET /api/model/info
```

**Response:**
```json
{
  "loaded": true,
  "device": "cpu",
  "classes": 8,
  "class_labels": [
    "unlabeled",
    "road",
    "building",
    "wall",
    "vegetation",
    "sky",
    "vehicle",
    "traffic"
  ]
}
```

### Predict

Perform inference on an uploaded image.

```
POST /api/predict
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form field `image` with the image file

**Response:**
```json
{
  "success": true,
  "prediction": "<base64-encoded-image>",
  "shape": [512, 1024, 3]
}
```

**Error Response:**
```json
{
  "error": "Error message"
}
```

## Image Processing Pipeline

1. **Upload**: Image is received as multipart form data
2. **Preprocessing**:
   - Convert to RGB
   - Resize (smallest side to 512px)
   - Center crop to 512x1024
   - Normalize using ImageNet stats
   - Convert to tensor
3. **Inference**: Model prediction with torch.no_grad()
4. **Postprocessing**:
   - Apply softmax
   - Get argmax for class prediction
   - Map to RGB colors
   - Encode as base64 PNG

## Dependencies

- Flask 3.0.0 - Web framework
- flask-cors 4.0.0 - CORS support
- torch 2.1.0 - Deep learning framework
- torchvision 0.16.0 - Vision utilities
- numpy 1.24.3 - Numerical operations
- pillow 10.1.0 - Image processing
- scikit-image 0.22.0 - Image processing
- albumentations 1.3.1 - Image augmentation

## Project Structure

```
backend/
├── app.py              # Main Flask application
└── requirements.txt    # Python dependencies
```

## Troubleshooting

### Model Not Loading

If you see "Model not loaded" errors:

1. Check that `model.pt` exists in the expected location
2. Verify the model file is a valid PyTorch model
3. Check file permissions
4. Set `MODEL_PATH` environment variable explicitly

Example:
```bash
export MODEL_PATH=/absolute/path/to/model.pt
python app.py
```

### Import Errors

If you get import errors for project modules:

The backend automatically adds the parent directory to the Python path to import from the root project files (`inference.py`, `network.py`, etc.). If this fails:

1. Verify the project structure is correct
2. Run from the `backend/` directory
3. Check that parent directory contains required files

### CORS Issues

If frontend cannot connect:

1. Verify Flask-CORS is installed: `pip install flask-cors`
2. Check that CORS is initialized in app.py: `CORS(app)`
3. Verify frontend is using correct API URL

### Memory Issues

For large models or high traffic:

1. Use CPU inference by default (automatic)
2. For GPU: Modify `device = torch.device('cuda')`
3. Adjust Gunicorn workers: `gunicorn -w 2` (fewer workers)
4. Enable model quantization for smaller memory footprint

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests (when available)
pytest
```

### Adding New Endpoints

1. Define route in `app.py`:
```python
@app.route('/api/new-endpoint', methods=['GET'])
def new_endpoint():
    return jsonify({'result': 'success'})
```

2. Update documentation
3. Test with frontend or curl

## Security Considerations

- File upload size limits are browser/server defaults
- No authentication implemented (add for production)
- CORS is open to all origins (restrict for production)
- No rate limiting (add for production)

## Performance Optimization

- Model loaded once on startup (not per request)
- Uses `torch.no_grad()` for inference
- CPU inference by default (change to GPU if available)
- Consider model quantization for faster inference

## License

University project for PA228 course.

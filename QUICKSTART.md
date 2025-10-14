# PA228 Frontend Quick Start Guide

This guide will help you get the frontend up and running quickly.

## What Was Added

✅ **Backend API** (`backend/`):
- Flask REST API for model inference
- CORS enabled for cross-origin requests
- Three endpoints: `/api/health`, `/api/model/info`, `/api/predict`
- Automatic model loading on startup

✅ **Frontend Application** (`frontend/`):
- Modern React 19 with Vite
- Three custom components:
  - `ImageUpload`: Drag-and-drop file upload
  - `PredictionDisplay`: Side-by-side image comparison
  - `ModelInfo`: Real-time backend status
- Responsive design with light/dark mode
- Professional UI with gradient header

## Prerequisites

- Python 3.8+ with PyTorch
- Node.js 18+ with npm 9+
- A trained `model.pt` file (place in project root)

## Quick Start (5 minutes)

### Terminal 1: Start Backend

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start Flask server (ensure model.pt exists in project root)
cd ..
python backend/app.py
```

Backend will run on: `http://localhost:5000`

### Terminal 2: Start Frontend

```bash
# Install frontend dependencies
cd frontend
npm install

# Start Vite dev server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

### Use the Application

1. Open `http://localhost:5173` in your browser
2. You should see the model status (green = ready, red = not available)
3. Drag and drop an image or click "Browse Files"
4. View the segmentation results side-by-side with the original

## Project Structure

```
PA228/
├── backend/                      # Flask API server
│   ├── app.py                   # Main Flask application
│   ├── requirements.txt         # Python dependencies
│   └── README.md               # Backend documentation
│
├── frontend/                    # React web application  
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── PredictionDisplay.jsx
│   │   │   └── ModelInfo.jsx
│   │   ├── App.jsx             # Main app
│   │   └── main.jsx            # Entry point
│   ├── package.json            # Node dependencies
│   └── README.md              # Frontend documentation
│
├── network.py                  # PyTorch model architecture
├── dataset.py                  # Dataset handling
├── training.py                 # Model training
├── inference.py                # Model inference
└── README_FRONTEND.md         # Complete documentation
```

## Configuration

### Backend Configuration

Set environment variables:
```bash
export MODEL_PATH=/path/to/model.pt  # Default: ./model.pt
export PORT=5000                      # Default: 5000
```

### Frontend Configuration

Edit `frontend/.env`:
```
VITE_API_URL=http://localhost:5000
```

## Troubleshooting

### "Backend not available" Error

**Problem**: Red warning in frontend showing backend is not available.

**Solutions**:
1. Ensure Flask backend is running (`python backend/app.py`)
2. Check backend console for errors
3. Verify `VITE_API_URL` in `frontend/.env` matches backend URL

### "Model not loaded" Error

**Problem**: Backend runs but model isn't loaded.

**Solutions**:
1. Verify `model.pt` exists in project root
2. Check file permissions
3. Set `MODEL_PATH` environment variable explicitly
4. Check backend logs for loading errors

### Port Already in Use

**Backend**: Change port with `export PORT=8000`
**Frontend**: Vite will prompt for an alternative port automatically

### Module Import Errors (Backend)

**Problem**: Python can't find `inference` or `network` modules.

**Solution**: The backend automatically adds the parent directory to Python path. Ensure:
- You run from the correct directory structure
- Parent directory contains `inference.py`, `network.py`, etc.

### Build Errors (Frontend)

**Problem**: `npm install` or `npm run build` fails.

**Solutions**:
1. Delete `node_modules/` and `package-lock.json`
2. Run `npm install` again
3. Ensure Node.js version is 18+

## Production Deployment

### Backend (Production)

```bash
cd backend
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (Production)

```bash
cd frontend
npm run build
# Serve the dist/ directory with nginx, Apache, etc.
```

## Features

- 🖼️ **Drag & Drop**: Easy image upload
- 🔄 **Real-time Status**: See backend connection status
- 🎨 **Visual Results**: Side-by-side comparison
- 📊 **Class Legend**: See all 8 segmentation classes
- 🌓 **Dark/Light Mode**: Automatic theme detection
- 📱 **Responsive**: Works on all devices

## API Endpoints

### Health Check
```
GET http://localhost:5000/api/health
Response: {"status": "ok", "model_loaded": true}
```

### Model Info
```
GET http://localhost:5000/api/model/info
Response: {"loaded": true, "device": "cpu", "classes": 8, ...}
```

### Predict
```
POST http://localhost:5000/api/predict
Body: multipart/form-data with 'image' file
Response: {"success": true, "prediction": "<base64>", ...}
```

## Next Steps

1. **Train a Model**: Use `training.py` to train on your dataset
2. **Test Inference**: Upload test images through the web interface
3. **Customize UI**: Edit components in `frontend/src/components/`
4. **Add Features**: Extend the API or add new UI components
5. **Deploy**: Follow production deployment steps above

## Support

- **Frontend Docs**: `frontend/README.md`
- **Backend Docs**: `backend/README.md`
- **Complete Guide**: `README_FRONTEND.md`

## Technology Stack

- **Backend**: Flask 3.0, PyTorch 2.1, Python 3.8+
- **Frontend**: React 19, Vite 7, JavaScript
- **Styling**: Modern CSS3 with responsive design
- **API**: REST with JSON and base64 image encoding

---

**Note**: This is a university project for the PA228 Computer Vision course. The frontend provides an easy-to-use interface for the existing PyTorch segmentation model.

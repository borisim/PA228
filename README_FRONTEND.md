# PA228 Image Segmentation Project

A complete image segmentation application with machine learning backend and React frontend interface.

## Project Overview

This project implements semantic segmentation for urban scene understanding, with a PyTorch-based neural network backend and a modern web-based frontend for easy interaction.

## Project Structure

```
PA228/
├── backend/                # Flask API server
│   ├── app.py             # Main Flask application
│   └── requirements.txt   # Python dependencies
├── frontend/              # React web application
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── App.jsx        # Main app component
│   │   └── ...
│   ├── package.json       # Node dependencies
│   └── README.md          # Frontend documentation
├── dataset.py             # Dataset handling
├── network.py             # Neural network architecture
├── training.py            # Model training script
├── inference.py           # Model inference script
├── evaluation.py          # Evaluation metrics
└── README_FRONTEND.md     # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm 9+
- PyTorch 2.1.0+

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Ensure you have a trained model file (`model.pt`) in the project root or specify its path:
```bash
export MODEL_PATH=/path/to/model.pt
```

3. Start the Flask backend:
```bash
python app.py
```

The backend will run on `http://localhost:5000` by default.

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Configure the API URL (if different from default):
```bash
cp .env.example .env
# Edit .env to set VITE_API_URL=http://localhost:5000
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Usage

1. **Start Both Servers**: Make sure both backend (Flask) and frontend (Vite) servers are running
2. **Open Browser**: Navigate to `http://localhost:5173`
3. **Upload Image**: Drag and drop or browse to select an image
4. **View Results**: See the original image and segmentation prediction side-by-side

## Features

### Backend API (`backend/app.py`)

- **Flask REST API**: Exposes model inference endpoints
- **CORS Enabled**: Allows frontend communication
- **Model Management**: Automatic model loading on startup
- **Image Processing**: Handles image preprocessing and postprocessing

#### API Endpoints

- `GET /api/health` - Check server and model status
- `GET /api/model/info` - Get model configuration details
- `POST /api/predict` - Upload image and receive segmentation prediction

### Frontend Application (`frontend/`)

- **Modern React UI**: Built with React 19 and Vite
- **Drag & Drop Upload**: Easy image upload interface
- **Real-time Inference**: Instant segmentation results
- **Visual Comparison**: Side-by-side original and predicted images
- **Class Legend**: Color-coded segmentation classes
- **Responsive Design**: Works on desktop and mobile devices

## Model Training

To train a new model:

```bash
python training.py <path_to_dataset>
```

This will:
- Train the UNet model on the provided dataset
- Save the trained model as `model.pt`
- Generate learning curves (`learning_curves.png`)
- Save model architecture visualization (`model_architecture.png`)

## Model Inference (CLI)

For command-line inference without the web interface:

```bash
python inference.py <path_to_dataset> <path_to_model> [num_samples]
```

Results will be saved to the `output_predictions/` directory.

## Evaluation

To evaluate model performance:

```bash
python evaluation.py SEG <path_to_ground_truth> <path_to_predictions>
```

## Segmentation Classes

The model predicts 8 semantic classes:

| Class       | RGB Color         | Description          |
|-------------|-------------------|----------------------|
| Unlabeled   | (0, 0, 0)         | Black                |
| Road        | (128, 64, 128)    | Purple               |
| Building    | (70, 70, 70)      | Dark Gray            |
| Wall        | (153, 153, 153)   | Light Gray           |
| Vegetation  | (107, 142, 35)    | Olive Green          |
| Sky         | (70, 130, 180)    | Steel Blue           |
| Vehicle     | (220, 20, 60)     | Crimson              |
| Traffic     | (0, 0, 142)       | Dark Blue            |

## Development

### Frontend Development

```bash
cd frontend
npm run dev      # Start dev server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run linter
```

### Backend Development

```bash
cd backend
python app.py    # Start in debug mode
# Set PORT environment variable to change port
export PORT=8000
python app.py
```

## Production Deployment

### Backend

1. Use a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

2. Set environment variables:
```bash
export MODEL_PATH=/path/to/model.pt
export PORT=5000
```

### Frontend

1. Build the production bundle:
```bash
cd frontend
npm run build
```

2. Serve the `dist/` directory with a web server (nginx, Apache, etc.)

3. Update API URL in production environment:
```bash
VITE_API_URL=https://your-api-domain.com
```

## Troubleshooting

### Backend Issues

- **Model not found**: Ensure `model.pt` exists or set `MODEL_PATH` environment variable
- **Import errors**: Verify all Python dependencies are installed
- **CORS errors**: Check Flask-CORS is installed and configured

### Frontend Issues

- **Cannot connect to backend**: Verify backend is running and `VITE_API_URL` is correct
- **Build errors**: Delete `node_modules` and run `npm install` again
- **Port conflicts**: Change the port in `vite.config.js`

## Contributing

1. Follow existing code style and conventions
2. Test changes thoroughly before committing
3. Update documentation for new features
4. Ensure backward compatibility

## License

This is a university project for the PA228 Computer Vision course.

## Acknowledgments

- Course: PA228 - Computer Vision
- Dataset: Urban street scenes for semantic segmentation
- Framework: PyTorch for deep learning, React for UI

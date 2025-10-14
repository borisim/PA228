# PA228 Image Segmentation - Frontend

This is the frontend application for the PA228 image segmentation project. It provides a user-friendly interface to upload images and visualize semantic segmentation results.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or browse to upload images
- 🤖 **Real-time Inference**: Get segmentation predictions from the trained model
- 🎨 **Visual Comparison**: View original and segmented images side-by-side
- 📊 **Class Legend**: Clear visualization of all segmentation classes
- 🔄 **Model Status**: Real-time backend connection monitoring

## Tech Stack

- **Framework**: React 19.1.1
- **Build Tool**: Vite 7.1.7
- **Styling**: CSS3 with CSS Modules
- **API Communication**: Fetch API

## Prerequisites

- Node.js 18.x or higher
- npm 9.x or higher
- Backend API server running (see Backend Setup section)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure the API URL (optional):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your backend URL (default is http://localhost:5000)
VITE_API_URL=http://localhost:5000
```

## Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173` (default Vite port).

### Development Features

- Hot Module Replacement (HMR) for instant updates
- ESLint for code quality
- React Fast Refresh for component updates

## Building for Production

Create an optimized production build:
```bash
npm run build
```

The built files will be in the `dist` directory.

Preview the production build locally:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # React components
│   │   ├── ImageUpload.jsx       # Image upload component
│   │   ├── ImageUpload.css
│   │   ├── PredictionDisplay.jsx # Results display component
│   │   ├── PredictionDisplay.css
│   │   ├── ModelInfo.jsx         # Model status component
│   │   └── ModelInfo.css
│   ├── App.jsx          # Main application component
│   ├── App.css          # Application styles
│   ├── main.jsx         # Application entry point
│   └── index.css        # Global styles
├── .env                 # Environment variables
├── .env.example         # Example environment variables
├── package.json         # Dependencies and scripts
└── vite.config.js       # Vite configuration
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The frontend communicates with the backend API through the following endpoints:

- `GET /api/health` - Check backend health status
- `GET /api/model/info` - Get model information
- `POST /api/predict` - Upload image and get segmentation prediction

### API Request Example

```javascript
const formData = new FormData()
formData.append('image', imageFile)

const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData,
})

const data = await response.json()
// data.prediction contains base64-encoded prediction image
```

## Segmentation Classes

The model predicts 8 different classes:

1. **Unlabeled** - RGB(0, 0, 0) - Black
2. **Road** - RGB(128, 64, 128) - Purple
3. **Building** - RGB(70, 70, 70) - Dark Gray
4. **Wall** - RGB(153, 153, 153) - Light Gray
5. **Vegetation** - RGB(107, 142, 35) - Olive Green
6. **Sky** - RGB(70, 130, 180) - Steel Blue
7. **Vehicle** - RGB(220, 20, 60) - Crimson
8. **Traffic** - RGB(0, 0, 142) - Dark Blue

## Troubleshooting

### Backend Not Available

If you see "Backend not available" message:
1. Ensure the backend server is running on the configured port
2. Check that the `VITE_API_URL` in `.env` matches your backend URL
3. Verify CORS is enabled on the backend

### Model Not Loaded

If predictions fail with "Model not loaded":
1. Ensure the `model.pt` file exists in the backend directory
2. Check backend logs for model loading errors
3. Verify the model file is a valid PyTorch model

### Build Errors

If you encounter build errors:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Ensure you're using a compatible Node.js version

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Contributing

When contributing to the frontend:

1. Follow the existing code style
2. Use functional components with hooks
3. Add CSS modules for component styling
4. Test with both light and dark color schemes
5. Ensure responsive design works on mobile devices

## License

This is a university project for PA228 course.

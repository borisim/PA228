import { useState } from 'react'
import './App.css'
import ImageUpload from './components/ImageUpload'
import PredictionDisplay from './components/PredictionDisplay'
import ModelInfo from './components/ModelInfo'

function App() {
  const [originalImage, setOriginalImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

  const handleImageUpload = async (file) => {
    setLoading(true)
    setError(null)
    setPrediction(null)

    // Display original image
    const reader = new FileReader()
    reader.onload = (e) => setOriginalImage(e.target.result)
    reader.readAsDataURL(file)

    // Send to backend for prediction
    const formData = new FormData()
    formData.append('image', file)

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setPrediction(`data:image/png;base64,${data.prediction}`)
    } catch (err) {
      setError(err.message || 'Failed to get prediction')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>PA228 Image Segmentation</h1>
        <p>Upload an image to perform semantic segmentation</p>
      </header>

      <main className="app-main">
        <ModelInfo apiUrl={API_URL} />
        
        <ImageUpload 
          onUpload={handleImageUpload}
          loading={loading}
        />

        {error && (
          <div className="error-message">
            <p>Error: {error}</p>
          </div>
        )}

        {(originalImage || prediction) && (
          <PredictionDisplay
            originalImage={originalImage}
            prediction={prediction}
            loading={loading}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>PA228 Computer Vision Project</p>
      </footer>
    </div>
  )
}

export default App

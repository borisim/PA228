import './PredictionDisplay.css'

function PredictionDisplay({ originalImage, prediction, loading }) {
  return (
    <div className="prediction-display">
      <h2>Results</h2>
      
      <div className="images-container">
        <div className="image-box">
          <h3>Original Image</h3>
          <div className="image-wrapper">
            {originalImage ? (
              <img src={originalImage} alt="Original" />
            ) : (
              <div className="image-placeholder">No image</div>
            )}
          </div>
        </div>

        <div className="image-box">
          <h3>Segmentation Result</h3>
          <div className="image-wrapper">
            {loading ? (
              <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Processing...</p>
              </div>
            ) : prediction ? (
              <img src={prediction} alt="Prediction" />
            ) : (
              <div className="image-placeholder">Waiting for prediction...</div>
            )}
          </div>
        </div>
      </div>

      <div className="legend">
        <h3>Class Legend</h3>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(0, 0, 0)' }}></span>
            <span>Unlabeled</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(128, 64, 128)' }}></span>
            <span>Road</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(70, 70, 70)' }}></span>
            <span>Building</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(153, 153, 153)' }}></span>
            <span>Wall</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(107, 142, 35)' }}></span>
            <span>Vegetation</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(70, 130, 180)' }}></span>
            <span>Sky</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(220, 20, 60)' }}></span>
            <span>Vehicle</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: 'rgb(0, 0, 142)' }}></span>
            <span>Traffic</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PredictionDisplay

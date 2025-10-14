import { useRef } from 'react'
import './ImageUpload.css'

function ImageUpload({ onUpload, loading }) {
  const fileInputRef = useRef(null)

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file')
        return
      }
      onUpload(file)
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    event.currentTarget.classList.add('drag-over')
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    event.currentTarget.classList.remove('drag-over')
  }

  const handleDrop = (event) => {
    event.preventDefault()
    event.currentTarget.classList.remove('drag-over')
    
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      onUpload(file)
    } else {
      alert('Please drop an image file')
    }
  }

  return (
    <div className="image-upload">
      <div
        className="upload-area"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        
        <div className="upload-content">
          <svg
            className="upload-icon"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          
          <p className="upload-text">
            Drag and drop an image here, or
          </p>
          
          <button
            onClick={handleButtonClick}
            disabled={loading}
            className="upload-button"
          >
            {loading ? 'Processing...' : 'Browse Files'}
          </button>
          
          <p className="upload-hint">
            Supported formats: PNG, JPG, JPEG
          </p>
        </div>
      </div>
    </div>
  )
}

export default ImageUpload

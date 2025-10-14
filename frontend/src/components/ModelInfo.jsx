import { useState, useEffect } from 'react'
import './ModelInfo.css'

function ModelInfo({ apiUrl }) {
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/model/info`)
        if (response.ok) {
          const data = await response.json()
          setModelInfo(data)
        } else {
          setError('Model not loaded')
        }
      } catch {
        setError('Backend not available')
      } finally {
        setLoading(false)
      }
    }

    fetchModelInfo()
  }, [apiUrl])

  if (loading) {
    return (
      <div className="model-info loading">
        <p>Checking model status...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="model-info error">
        <p>⚠️ {error}</p>
        <p className="hint">Make sure the backend server is running</p>
      </div>
    )
  }

  return (
    <div className="model-info success">
      <div className="info-header">
        <span className="status-indicator">●</span>
        <span>Model Ready</span>
      </div>
      <div className="info-details">
        <span>Device: {modelInfo?.device}</span>
        <span>Classes: {modelInfo?.classes}</span>
      </div>
    </div>
  )
}

export default ModelInfo

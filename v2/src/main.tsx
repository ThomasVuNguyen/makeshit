import { StrictMode, Component } from 'react'
import type { ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'

// Minimal test first
console.log('main.tsx starting...')

// Error boundary to catch React errors
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean, error: Error | null }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: any) {
    console.error('React Error Boundary caught:', error)
    console.error('Component stack:', info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 20, color: 'red', fontFamily: 'monospace', background: '#fff' }}>
          <h1>Something went wrong</h1>
          <pre>{this.state.error?.message}</pre>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{this.state.error?.stack}</pre>
        </div>
      )
    }
    return this.props.children
  }
}

// Lazy load App to catch any import errors
const App = await import('./App.tsx').then(m => m.default).catch(e => {
  console.error('Failed to import App:', e)
  return () => <div style={{ color: 'red', padding: 20 }}>Failed to load App: {String(e)}</div>
})

try {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>,
  )
  console.log('React mounted successfully')
} catch (e) {
  console.error('Failed to mount React:', e)
  document.body.innerHTML = '<pre style="color:red;padding:20px">' + String(e) + '</pre>'
}

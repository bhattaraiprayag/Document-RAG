import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { FileUpload } from './components/FileUpload'
import { FileList } from './components/FileList'
import { ChatWindow } from './components/ChatWindow'

const queryClient = new QueryClient()

function App() {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <h1 className="text-3xl font-bold text-white">Hybrid RAG System</h1>
            <p className="text-sm text-gray-400 mt-1">
              Local-first document Q&A with parent-child chunking
            </p>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: File Management */}
            <div className="lg:col-span-1 space-y-4">
              <FileUpload />
              <FileList
                selectedFiles={selectedFiles}
                onSelectionChange={setSelectedFiles}
              />
            </div>

            {/* Right Column: Chat */}
            <div className="lg:col-span-2">
              <ChatWindow selectedFiles={selectedFiles} />
            </div>
          </div>
        </div>
      </div>
    </QueryClientProvider>
  )
}

export default App

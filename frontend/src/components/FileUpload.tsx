import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload } from 'lucide-react'
import { api } from '../api/client'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

export function FileUpload() {
  const [dragActive, setDragActive] = useState(false)
  const queryClient = useQueryClient()

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadMutation.mutate(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadMutation.mutate(e.target.files[0])
    }
  }

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <CardTitle className="text-lg text-white">Upload Document</CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${dragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-blue-400'
            }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('fileInput')?.click()}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-3" />
          <p className="text-sm text-gray-200 mb-1">
            {uploadMutation.isPending ? 'Uploading...' : 'Drop file here or click to browse'}
          </p>
          <p className="text-xs text-gray-400">PDF, DOCX, EPUB, MD, TXT</p>
          <input
            id="fileInput"
            type="file"
            className="hidden"
            accept=".pdf,.doc,.docx,.epub,.md,.txt"
            onChange={handleChange}
          />
        </div>
        {uploadMutation.isError && (
          <p className="text-sm text-red-500 mt-2">Upload failed. Please try again.</p>
        )}
        {uploadMutation.isSuccess && (
          <p className="text-sm text-green-600 mt-2">File uploaded successfully!</p>
        )}
      </CardContent>
    </Card>
  )
}

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Trash2, FileText } from 'lucide-react'
import { api } from '../api/client'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import * as Checkbox from '@radix-ui/react-checkbox'

interface FileListProps {
  selectedFiles: string[]
  onSelectionChange: (files: string[]) => void
}

export function FileList({ selectedFiles, onSelectionChange }: FileListProps) {
  const queryClient = useQueryClient()

  const { data: documents = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: api.getDocuments,
    refetchInterval: 3000,
  })

  const deleteMutation = useMutation({
    mutationFn: (fileHash: string) => api.deleteDocument(fileHash),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const toggleFile = (fileHash: string) => {
    if (selectedFiles.includes(fileHash)) {
      onSelectionChange(selectedFiles.filter((h) => h !== fileHash))
    } else {
      onSelectionChange([...selectedFiles, fileHash])
    }
  }

  const toggleAll = () => {
    if (selectedFiles.length === documents.length) {
      onSelectionChange([])
    } else {
      onSelectionChange(documents.map((d) => d.file_hash))
    }
  }

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="text-lg text-white">Documents ({documents.length})</CardTitle>
        {documents.length > 0 && (
          <Button variant="outline" size="sm" onClick={toggleAll} className="text-xs">
            {selectedFiles.length === documents.length ? 'Deselect All' : 'Select All'}
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <p className="text-sm text-gray-400">Loading...</p>
        ) : documents.length === 0 ? (
          <p className="text-sm text-gray-400">No documents uploaded yet</p>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.file_hash}
                className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all"
              >
                <Checkbox.Root
                  checked={selectedFiles.includes(doc.file_hash)}
                  onCheckedChange={() => toggleFile(doc.file_hash)}
                  className="h-5 w-5 shrink-0 rounded border-2 border-gray-500 bg-gray-800 data-[state=checked]:bg-blue-500 data-[state=checked]:border-blue-500 flex items-center justify-center cursor-pointer hover:border-blue-400 transition-colors"
                >
                  <Checkbox.Indicator className="text-white font-bold">✓</Checkbox.Indicator>
                </Checkbox.Root>
                <FileText className="h-4 w-4 text-gray-400 shrink-0" />
                <span className="text-sm font-medium text-white truncate flex-1 min-w-0" title={doc.file_name}>
                  {doc.file_name}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteMutation.mutate(doc.file_hash)
                  }}
                  disabled={deleteMutation.isPending}
                  className="shrink-0 hover:bg-red-500/20 hover:text-red-400 transition-all p-2 rounded-md"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

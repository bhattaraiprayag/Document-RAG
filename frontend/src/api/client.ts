import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000/api'

export type FileInfo = {
  file_hash: string
  file_name: string
}

export type IngestionStatus = {
  file_hash: string
  file_name: string
  stage: string
  progress: number
  error?: string
}

export const api = {
  async uploadDocument(file: File): Promise<{ file_hash: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await axios.post(`${API_BASE_URL}/documents/upload`, formData);
    return data;
  },

  async getDocuments(): Promise<FileInfo[]> {
    const { data } = await axios.get(`${API_BASE_URL}/documents`);
    return data;
  },

  async getIngestionStatus(fileHash: string): Promise<IngestionStatus> {
    const { data } = await axios.get(`${API_BASE_URL}/documents/status/${fileHash}`);
    return data;
  },

  async deleteDocument(fileHash: string): Promise<void> {
    await axios.delete(`${API_BASE_URL}/documents/${fileHash}`);
  },

  streamChat(query: string, selectedFiles?: string[]) {
    return fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, selected_files: selectedFiles }),
    });
  },
};

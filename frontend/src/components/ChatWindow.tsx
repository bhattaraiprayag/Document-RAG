import { useState, useRef, useEffect, type ComponentPropsWithoutRef } from 'react'
import { Send } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api/client'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatWindowProps {
  selectedFiles: string[]
}

export function ChatWindow({ selectedFiles }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isStreaming) return

    const userMessage: Message = { role: 'user', content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)

    try {
      const response = await api.streamChat(input, selectedFiles.length > 0 ? selectedFiles : undefined)

      if (!response.ok || !response.body) {
        throw new Error('Stream failed')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantMessage = ''

      // Add empty assistant message that will be updated
      setMessages((prev) => [...prev, { role: 'assistant', content: '' }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') break

            try {
              const json = JSON.parse(data)
              if (json.error) {
                setMessages((prev) => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: `**Error:** ${json.error}`,
                  }
                  return updated
                })
                return
              }
              if (json.token) {
                assistantMessage += json.token
                setMessages((prev) => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantMessage,
                  }
                  return updated
                })
              }
            } catch (e) {
              // Skip malformed JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error: Failed to get response. Please try again.' },
      ])
    } finally {
      setIsStreaming(false)
    }
  }

  return (
    <Card className="h-[calc(100vh-200px)] flex flex-col bg-gray-800 border-gray-700 overflow-hidden">
      <CardHeader className="border-b border-gray-700 flex-shrink-0">
        <CardTitle className="text-lg text-white">Chat</CardTitle>
        {selectedFiles.length > 0 && (
          <p className="text-sm text-blue-400">{selectedFiles.length} document(s) selected</p>
        )}
      </CardHeader>
      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-900 min-h-0">
          {messages.length === 0 ? (
            <div className="text-center text-gray-400 mt-8">
              <p>Upload documents and ask questions!</p>
              <p className="text-sm mt-2">Select documents from the list to query specific files</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-100'
                    }`}
                >
                  {msg.role === 'user' ? (
                    <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                  ) : (
                    <div className="prose prose-invert prose-sm max-w-none text-sm">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          // Customize code blocks
                          // eslint-disable-next-line @typescript-eslint/no-unused-vars
                          code({ inline, className, children, ...props }: ComponentPropsWithoutRef<'code'> & { inline?: boolean }) {
                            return inline ? (
                              <code className={`bg-gray-800 px-1.5 py-0.5 rounded text-xs font-mono text-blue-300 ${className || ''}`} {...props}>
                                {children}
                              </code>
                            ) : (
                              <code className={`block bg-gray-800 p-3 rounded-md text-xs font-mono overflow-x-auto ${className || ''}`} {...props}>
                                {children}
                              </code>
                            )
                          },
                          // Customize links
                          a({ children, href, ...props }: ComponentPropsWithoutRef<'a'>) {
                            return (
                              <a href={href} className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer" {...props}>
                                {children}
                              </a>
                            )
                          },
                          // Customize lists
                          ul({ children, ...props }: ComponentPropsWithoutRef<'ul'>) {
                            return <ul className="list-disc list-inside space-y-1 my-2" {...props}>{children}</ul>
                          },
                          ol({ children, ...props }: ComponentPropsWithoutRef<'ol'>) {
                            return <ol className="list-decimal list-inside space-y-1 my-2" {...props}>{children}</ol>
                          },
                          // Customize paragraphs
                          p({ children, ...props }: ComponentPropsWithoutRef<'p'>) {
                            return <p className="mb-2 leading-relaxed" {...props}>{children}</p>
                          },
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="border-t border-gray-700 p-4 bg-gray-800">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400"
              disabled={isStreaming}
            />
            <Button type="submit" disabled={isStreaming || !input.trim()} className="bg-blue-600 hover:bg-blue-700">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}

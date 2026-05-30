import { useState, useCallback, useRef } from 'react'
import SearchBar from './components/SearchBar'
import StreamingAnswer from './components/StreamingAnswer'
import BookCard from './components/BookCard'
import BookModal from './components/BookModal'
import Balatro from './components/Balatro'
import styles from './App.module.css'

const MAX_RETRIES = 2

function extractBookReview(answerText, bookTitle) {
  if (!answerText || !bookTitle) return ''
  // Split on blank lines OR the start of a numbered/bulleted list item
  const blocks = answerText
    .split(/\n\n+|\n(?=\d+\.|\s*[-*]\s)/)
    .map(b => b.replace(/^\s*\d+\.\s+|^\s*[-*]\s+/, '').trim())
    .filter(Boolean)
  const lower = bookTitle.toLowerCase()
  const matched = blocks.filter(b => b.toLowerCase().includes(lower))
  if (!matched.length) return ''
  return matched[0]
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/`(.+?)`/g, '$1')
    .replace(/^#+\s+/gm, '')
    .trim()
}

/**
 * Reorder `sourcesArr` by the position each book's title first appears in
 * `answerText`. Books not found in the answer are pushed to the end.
 */
function reorderByMention(answerText, sourcesArr) {
  const lower = answerText.toLowerCase()
  return [...sourcesArr].sort((a, b) => {
    const posA = lower.indexOf((a.title || '').toLowerCase())
    const posB = lower.indexOf((b.title || '').toLowerCase())
    const rankA = posA === -1 ? Infinity : posA
    const rankB = posB === -1 ? Infinity : posB
    return rankA - rankB
  })
}

export default function App() {
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState(null)
  const [selectedBook, setSelectedBook] = useState(null) // { work_id, ...initialData }
  const retryCount = useRef(0)
  // Refs so we can read the latest values inside the streaming loop
  const answerRef = useRef('')
  const sourcesRef = useRef([])

  const runSearch = useCallback(async (query) => {
    setAnswer('')
    setSources([])
    setError(null)
    setStreaming(true)
    answerRef.current = ''
    sourcesRef.current = []

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let finished = false

      while (!finished) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue

          let msg
          try { msg = JSON.parse(raw) } catch { continue }

          if (msg.type === 'token') {
            answerRef.current += msg.content
            setAnswer(answerRef.current)
          } else if (msg.type === 'sources') {
            sourcesRef.current = msg.data
            setSources(msg.data)
          } else if (msg.type === 'error') {
            throw new Error(msg.message)
          } else if (msg.type === 'done') {
            if (msg.empty && retryCount.current < MAX_RETRIES) {
              // Silently retry without telling the user
              retryCount.current += 1
              setStreaming(false)
              runSearch(query)
              return
            }
            // Reorder source cards to match the order books appear in the answer
            const reordered = reorderByMention(answerRef.current, sourcesRef.current)
            setSources(reordered)
            finished = true
            break
          }
        }
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setStreaming(false)
    }
  }, [])

  const handleSearch = useCallback((query) => {
    retryCount.current = 0
    runSearch(query)
  }, [runSearch])

  return (
    <>
      <Balatro
        isRotate={false}
        mouseInteraction
        pixelFilter={745}
        color1="#DE443B"
        color2="#006BB4"
        color3="#162325"
      />
      <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.logo}>📚 Book Recommender</h1>
        <p className={styles.subtitle}>
          Powered by Goodreads · pgvector · Cohere Rerank · OpenRouter
        </p>
      </header>

      <main className={styles.main}>
        <SearchBar onSearch={handleSearch} loading={streaming} />

        {error && (
          <div className={styles.error}>
            ⚠️ {error}
          </div>
        )}

        {streaming && <StreamingAnswer text={answer} streaming={streaming} />}

        {!streaming && sources.length > 0 && (
          <section>
            <h2 className={styles.sourcesHeading}>Recommendations</h2>
            <div className={styles.list}>
              {sources.map((book, i) => (
                <BookCard
                  key={book.work_id ?? i}
                  book={book}
                  reviewText={extractBookReview(answer, book.title)}
                  onClick={() => setSelectedBook(book)}
                />
              ))}
            </div>
          </section>
        )}

        {!streaming && answer && sources.length === 0 && (
          <StreamingAnswer text={answer} streaming={false} />
        )}
      </main>

      <footer className={styles.footer}>
        Book Recommender &mdash; RAG with pgvector
      </footer>

      {/* Book detail modal */}
      {selectedBook && (
        <BookModal
          workId={selectedBook.work_id}
          initialBook={selectedBook}
          onClose={() => setSelectedBook(null)}
        />
      )}
      </div>
    </>
  )
}

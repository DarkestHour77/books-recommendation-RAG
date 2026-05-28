import { useEffect, useState, useCallback } from 'react'
import styles from './BookModal.module.css'

const PLACEHOLDER =
  'https://images.unsplash.com/photo-1512820790803-83ca734da794?w=240&h=360&fit=crop'

function StarBar({ label, count, total }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0
  return (
    <div className={styles.starRow}>
      <span className={styles.starLabel}>{label}</span>
      <div className={styles.starTrack}>
        <div className={styles.starFill} style={{ width: `${pct}%` }} />
      </div>
      <span className={styles.starCount}>{count?.toLocaleString() ?? '—'}</span>
    </div>
  )
}

export default function BookModal({ workId, initialBook, onClose }) {
  const [book, setBook] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Close on Escape key
  const handleKey = useCallback(
    (e) => { if (e.key === 'Escape') onClose() },
    [onClose]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', handleKey)
      document.body.style.overflow = ''
    }
  }, [handleKey])

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`${import.meta.env.VITE_API_BASE || ''}/api/book/${workId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => { setBook(data); setLoading(false) })
      .catch((err) => { setError(err.message); setLoading(false) })
  }, [workId])

  // Parse genres: could be JSON array string or comma-separated
  const parseGenres = (raw) => {
    if (!raw) return []
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) return parsed
    } catch {}
    return raw.split(',').map((g) => g.trim()).filter(Boolean)
  }

  const display = book || initialBook
  const genres = parseGenres(book?.genres)
  const totalRatings = book?.ratings_count || 0

  return (
    <div className={styles.backdrop} onClick={onClose} role="dialog" aria-modal="true">
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>

        {/* Close button */}
        <button className={styles.close} onClick={onClose} aria-label="Close">✕</button>

        {loading && (
          <div className={styles.loadingState}>
            <div className={styles.spinner} />
            <p>Loading book details…</p>
          </div>
        )}

        {error && !loading && (
          <div className={styles.errorState}>
            <p>⚠️ Could not load details: {error}</p>
          </div>
        )}

        {!loading && display && (
          <div className={styles.content}>
            {/* Left: cover + quick stats */}
            <div className={styles.sidebar}>
              <img
                className={styles.cover}
                src={display.image_url || PLACEHOLDER}
                alt={`Cover of ${display.title}`}
                onError={(e) => { e.currentTarget.src = PLACEHOLDER }}
              />

              {book?.avg_rating && (
                <div className={styles.ratingBig}>
                  <span className={styles.ratingNum}>
                    {Number(book.avg_rating).toFixed(2)}
                  </span>
                  <span className={styles.ratingStars}>
                    {'★'.repeat(Math.round(Number(book.avg_rating)))}
                    {'☆'.repeat(5 - Math.round(Number(book.avg_rating)))}
                  </span>
                  <span className={styles.ratingCount}>
                    {totalRatings.toLocaleString()} ratings
                  </span>
                </div>
              )}

              {book && (
                <div className={styles.starBars}>
                  <StarBar label="5★" count={book.star5_ratings} total={totalRatings} />
                  <StarBar label="4★" count={book.star4_ratings} total={totalRatings} />
                  <StarBar label="3★" count={book.star3_ratings} total={totalRatings} />
                  <StarBar label="2★" count={book.star2_ratings} total={totalRatings} />
                  <StarBar label="1★" count={book.star1_ratings} total={totalRatings} />
                </div>
              )}
            </div>

            {/* Right: main info */}
            <div className={styles.body}>
              <h2 className={styles.title}>{display.title}</h2>
              {display.author && (
                <p className={styles.author}>by {display.author}</p>
              )}

              {/* Genre chips */}
              {genres.length > 0 && (
                <div className={styles.genres}>
                  {genres.map((g) => (
                    <span key={g} className={styles.genre}>{g}</span>
                  ))}
                </div>
              )}

              {/* Meta grid */}
              <dl className={styles.meta}>
                {book?.original_publication_year && (
                  <>
                    <dt>Published</dt>
                    <dd>{book.original_publication_year}</dd>
                  </>
                )}
                {book?.num_pages && (
                  <>
                    <dt>Pages</dt>
                    <dd>{book.num_pages.toLocaleString()}</dd>
                  </>
                )}
                {book?.isbn && (
                  <>
                    <dt>ISBN</dt>
                    <dd>{book.isbn}</dd>
                  </>
                )}
                {book?.isbn13 && (
                  <>
                    <dt>ISBN-13</dt>
                    <dd>{book.isbn13}</dd>
                  </>
                )}
                {book?.reviews_count > 0 && (
                  <>
                    <dt>Reviews</dt>
                    <dd>{Number(book.reviews_count).toLocaleString()}</dd>
                  </>
                )}
              </dl>

              {/* Description */}
              {book?.description && (
                <div className={styles.descSection}>
                  <h3 className={styles.descHeading}>About this book</h3>
                  <p className={styles.description}>{book.description}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

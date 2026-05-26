import styles from './BookCard.module.css'

const PLACEHOLDER =
  'https://images.unsplash.com/photo-1512820790803-83ca734da794?w=120&h=180&fit=crop'

export default function BookCard({ book, onClick }) {
  const {
    title = 'Unknown Title',
    author = '',
    avg_rating,
    original_publication_year,
    image_url,
    work_id,
  } = book

  return (
    <article
      className={styles.card}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick?.() }}
      aria-label={`View details for ${title}`}
    >
      <img
        className={styles.cover}
        src={image_url || PLACEHOLDER}
        alt={`Cover of ${title}`}
        onError={(e) => { e.currentTarget.src = PLACEHOLDER }}
      />
      <div className={styles.info}>
        <p className={styles.title}>{title}</p>
        {author && <p className={styles.author}>{author}</p>}
        <div className={styles.meta}>
          {avg_rating && (
            <span className={styles.rating}>
              <span className={styles.star}>★</span>
              {Number(avg_rating).toFixed(1)}
            </span>
          )}
          {original_publication_year && (
            <span className={styles.year}>📅 {original_publication_year}</span>
          )}
        </div>
        <span className={styles.viewDetails}>View details →</span>
      </div>
    </article>
  )
}

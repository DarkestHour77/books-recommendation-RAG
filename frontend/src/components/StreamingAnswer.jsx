import ReactMarkdown from 'react-markdown'
import styles from './StreamingAnswer.module.css'

export default function StreamingAnswer({ text, streaming }) {
  if (!text && !streaming) return null

  return (
    <section className={styles.section}>
      <h2 className={styles.heading}>Answer</h2>
      <div className={styles.body}>
        <ReactMarkdown>{text}</ReactMarkdown>
        {streaming && <span className={styles.cursor}>▌</span>}
      </div>
    </section>
  )
}

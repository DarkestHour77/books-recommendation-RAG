import { useState } from 'react'
import styles from './SearchBar.module.css'

export default function SearchBar({ onSearch, loading }) {
  const [value, setValue] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (value.trim() && !loading) onSearch(value.trim())
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <input
        className={styles.input}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="e.g. sci-fi novels with strong female leads under 400 pages"
        disabled={loading}
        autoFocus
      />
      <button className={styles.button} type="submit" disabled={loading || !value.trim()}>
        {loading ? <Spinner /> : 'Search'}
      </button>
    </form>
  )
}

function Spinner() {
  return (
    <svg className={styles.spinner} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"
        strokeDasharray="31.4" strokeDashoffset="10" />
    </svg>
  )
}

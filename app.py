import os
import sys

import streamlit as st
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from scripts.query import run_query, setup_conn

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")


@st.cache_resource
def get_db_pool():
    pool = SimpleConnectionPool(minconn=1, maxconn=8, dsn=os.getenv("DATABASE_URL"))
    return pool


def get_conn():
    conn = get_db_pool().getconn()
    setup_conn(conn)
    return conn


def release_conn(conn):
    get_db_pool().putconn(conn)


st.title("📚 Book Recommender")
st.caption("Powered by Goodreads data · pgvector · Cohere Rerank · OpenRouter")

query = st.text_input(
    "What kind of book are you looking for?",
    placeholder="e.g. 'sci-fi novels with strong female leads under 400 pages'",
)

search = st.button("Search", type="primary")

if search and query.strip():
    conn = get_conn()
    try:
        with st.spinner("Searching..."):
            stream, sources = run_query(conn, query.strip())

        st.subheader("Answer")
        answer_placeholder = st.empty()
        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)

        if sources:
            st.subheader("Sources")
            cols = st.columns(min(len(sources), 3))
            for i, src in enumerate(sources):
                col = cols[i % 3]
                with col:
                    if src.get("image_url"):
                        st.image(src["image_url"], width=120)
                    st.markdown(f"**{src.get('title', 'Unknown')}** `{src.get('work_id', '')}`")
                    st.markdown(f"*{src.get('author', '')}*")
                    if src.get("avg_rating"):
                        st.markdown(f"⭐ {src['avg_rating']:.1f}")
                    if src.get("original_publication_year"):
                        st.markdown(f"📅 {src['original_publication_year']}")
                    st.markdown("---")
    finally:
        release_conn(conn)
elif search:
    st.warning("Please enter a query.")

# Book Recommendation System


## Ingetion Pipeline


### Assests
- In the assests folder there is goodreads_data_dictionary.csv, this file defines all the columns in the goodreads_reviews_deduplicated and goodreads_works.csv.
- In goodreads_merged.json there is all the detailed data about the books.
- I want the goodreads_merged.json as the Database for this project.

### Scripts
- In the scripts folder, create a document loader that reads the raw bytes and hands them to the right parser for the file type.
- Use LangChain's DirectoryLoader library for this. 
- It will auto detect formats and dispatch accordingly

- The data needs to divided into chunks using an AI model to detect top shifts and split there.
- Also add overlap of 10-20% of chunk size so context at chunk boundaries isn't lost.
- Each chunk also gets metadata attached which is critical for filtering later.

### Chunk text — gets embedded and semantically searched
- original_title
- author
- description
- genres
- review_text

### Metadata — filtering, sorting, display
- work_id          → traceability, linking reviews back to books
- isbn / isbn13    → external lookup
- original_publication_year  → filter by era ("books from the 90s")
- num_pages        → filter ("short reads only")
- image_url        → display in results
- avg_rating       → sort by quality
- ratings_count    → sort by popularity
- review_id        → traceability
- user_id          → filter by user ("show me my reviews")
- started_at / read_at / date_added  → filter by time
- rating           → filter ("only 5 star reviews")


- Each chunk is passed through an embedding model that converts the text into a dense vector (an array of floating-point numbers). 
- The key constraint: you must use the same model at query time that you used here, otherwise the vector spaces won't align. 

- The final vectors (along with the original chunk text and metadata) are written to a vector database
- The store builds an index (typically HNSW — a graph-based structure) that makes approximate nearest-neighbour search fast at query time

## Query Pipeline


### User Query
- Take user input from console.
- Do basic cleaning of the prompt.
- Run spelling correction on the prompt.
- Rewrite the user query with this method,
rewrite_prompt = f"""
Rewrite this user query to be more explicit and retrieval-friendly.
Keep it concise. Return only the rewritten query.

Query: {user_query}
"""

- Then extract Metadata from the query, This way the semantic search focuses on meaning, and the hard constraints are handled separately by metadata
 filtering 
- The query gets passed through an embedding model — critically, the exact same model you used during ingestion. 
- This converts the question into a vector in the same vector space as your stored chunks.

- The query vector gets sent to the vector store, which runs an HNSW search and returns the top-k most similar chunks — typically k=5 to 20. 
- This is also where metadata filters kick in. 
- For your book dataset, a query like "5 star sci-fi books" would combine the semantic similarity search with a metadata filter like rating = 5 and 
genres contains sci-fi, so you're not just finding semantically similar chunks but also satisfying hard constraints.
- Then a reranker takes those k chunks and the original query and scores them more carefully using a cross-encoder model. 
- it looks at the query and each chunk together rather than independently. 
- The result is a much more accurate relevance ranking.
- then trim down to the top 3–5 chunks before passing to the LLM.

### LLM
- Assemble everything into a prompt for the LLM, the structure should be something like this 
System: You are a helpful assistant. Answer only using the provided context.
        If the answer isn't in the context, say you don't know.

Context:
  [chunk 1 text]
  [chunk 2 text]
  [chunk 3 text]

Question: {user_query}
- then this assemble prompt goes to the LLM that generates the final answer.
- The LLM's job here is synthesis — taking the retrieved chunks and writing a coherent, fluent answer

### Response
- The answer goes back to the user along with source references like original_title, work_id, author, etc which should be available in the metadata for 
the chunks loaded into the LLM.


Any AI used in here, call it through a variable defined in .env file.

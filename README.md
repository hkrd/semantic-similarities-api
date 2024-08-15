# Semantic Similarities API


This API exposes endpoint to insert strings as embeddings in a local database
and an endpoint to search for the most similar embeddings given a query.
The service uses Cosine Similarity to search for Embeddings

# Endpoints

`/insert` - inserts a string as an mebedding to the local embeddings database
`/similarities` - searches for the 5 most similar embeddings given a query

# Install
`poetry install`

# Run
`./run.sh`

# Test
`pytest tests`
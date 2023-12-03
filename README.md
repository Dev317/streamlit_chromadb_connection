# 📂 ChromaDBConnection

Connection for Chroma vector database, `ChromaDBConnection`, has been released which makes it easy to connect any Streamlit LLM-powered app to.

With `st.connection()`, connecting to a Chroma vector database becomes just a few lines of code:


```python
import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

configuration = {
    "client": "PersistentClient",
    "path": "/tmp/.chroma"
}

collection_name = "documents_collection"

conn = st.connection("chromadb",
                    type=ChromaDBConnection,
                    **configuration)
documents_collection_df = conn.get_collection_data(collection_name)
st.dataframe(documents_collection_df)
```

## 📑 ChromaDBConnection API

### _connect()
There are 2 ways to connect to a Chroma client:
1. **PersistentClient**: Data will be persisted to a local machine
    ```python
    import streamlit as st
    from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

    configuration = {
        "client": "PersistentClient",
        "path": "/tmp/.chroma"
    }

    conn = st.connection(name="persistent_chromadb",
                        type=ChromadbConnection,
                        **configuration)
    ```

2. **HttpClient**: Data will be persisted to a cloud server where Chroma resides
    ```python
    import streamlit as st
    from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

    configuration = {
        "client": "HttpClient",
        "host": "localhost",
        "port": 8000,
    }

    conn = st.connection(name="http_connection",
                         type=ChromadbConnection,
                         **configuration)
    ```


### create_collection()
In order to create a Chroma collection, one needs to supply a `collection_name` and `embedding_function_name`, `embedding_config` and (optional) `metadata`.

There are current possible options for `embedding_function_name`:
- DefaultEmbeddingFunction
- SentenceTransformerEmbeddingFunction
- OpenAIEmbeddingFunction
- CohereEmbeddingFunction
- GooglePalmEmbeddingFunction
- GoogleVertexEmbeddingFunction
- HuggingFaceEmbeddingFunction
- InstructorEmbeddingFunction
- Text2VecEmbeddingFunction
- ONNXMiniLM_L6_V2

For `DefaultEmbeddingFunction`, the `embedding_config` argument can be left as an empty string. However, for other embedding functions such as `OpenAIEmbeddingFunction`, one needs to provide configuration such as:

```python
embedding_config = {
    api_key: "{OPENAI_API_KEY}",
    model_name: "{OPENAI_MODEL}",
}
```

One can also change the distance function by changing the `metadata` argument, such as:

```python
metadata = {"hnsw:space": "l2"} # Squared L2 norm
metadata = {"hnsw:space": "cosine"} # Cosine similarity
metadata = {"hnsw:space": "ip"} # Inner product
```

Sample code to create connection:

```python
collection_name = "documents_collection"
embedding_function_name = "DefaultEmbeddingFunction"
conn.create_collection(collection_name=collection_name,
                       embedding_function_name=embedding_function_name,
                       embedding_config={},
                       metadata = {"hnsw:space": "cosine"})
```

### get_collection_data()
This method returns a dataframe that consists of the embeddings and documents of a collection.
The `attributes` argument is a list of attributes to be included in the DataFrame.
The following code snippet will return all data in a collection in the form of a DataFrame, with 2 columns: `documents` and `embeddings`.

```python
collection_name = "documents_collection"
conn.get_collection_data(collection_name=collection_name,
                        attributes= ["documents", "embeddings"])
```

### delete_collection()
This method deletes the stated collection name.

```python
collection_name = "documents_collection"
conn.delete_collection(collection_name=collection_name)
```

### upload_document()
This method uploads documents to a collection.
If embeddings are not provided, the method will embed the documents using the embedding function specified in the collection.


```python
collection_name = "documents_collection"
conn.upload_document(collection_name=collection_name,
                     documents=["lorem ipsum", "doc2", "doc3"],
                     metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                     ids=["id1", "id2", "id3"],
                     embeddings=None)
```

### query()
This method retrieves top k relevant document based on a list of queries supplied.
The result will be in a dataframe where each row will shows the top k relevant documents of each query.

```python
collection_name = "documents_collection"
conn.upload_document(collection_name=collection_name,
                     documents=["lorem ipsum", "doc2", "doc3"],
                     metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                     ids=["id1", "id2", "id3"],
                     embeddings=None)

queried_data = conn.query(collection_name=collection_name,
                          query=["random_query1", "random_query2"],
                          num_results_limit=10,
                          attributes=["documents", "embeddings", "metadatas", "data"])
```

Metadata and document filters are also provided in `where_metadata_filter` and `where_document_filter` arguments respectively for more relevant search. For better understanding on the usage of where filters, please refer to: https://docs.trychroma.com/usage-guide#using-where-filters

```python
queried_data = conn.query(collection_name=collection_name,
                         query=["this is"],
                         num_results_limit=10,
                         attributes=["documents", "embeddings", "metadatas", "data"],
                         where_metadata_filter={"chapter": "3"})
```


***
🎉 That's it! `ChromaDBConnection` is ready to be used with `st.connection()`. 🎉
***

## Contribution 🔥
```
author={Vu Quang Minh},
github={Dev317},
year={2023}
```

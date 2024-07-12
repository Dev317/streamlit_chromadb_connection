import os
import streamlit
from streamlit.connections import BaseConnection
import chromadb
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    CohereEmbeddingFunction,
    GooglePalmEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    InstructorEmbeddingFunction,
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
    Text2VecEmbeddingFunction,
    ONNXMiniLM_L6_V2,
    GoogleVertexEmbeddingFunction
)
from chromadb.api import ClientAPI
from chromadb import PersistentClient, HttpClient
from chromadb.api.types import Documents, EmbeddingFunction
from chromadb.api.models.Collection import Collection
from typing import Dict, List, Union, cast
from typing_extensions import override
import pandas as pd


class ChromadbConnection(BaseConnection):
    
    EMBEDDING_FUNCTION_MAP = {
        "DefaultEmbeddingFunction": DefaultEmbeddingFunction,
        "SentenceTransformerEmbeddingFunction": SentenceTransformerEmbeddingFunction,
        "OpenAIEmbeddingFunction": OpenAIEmbeddingFunction,
        "CohereEmbeddingFunction": CohereEmbeddingFunction,
        "GooglePalmEmbeddingFunction": GooglePalmEmbeddingFunction,
        "GoogleVertexEmbeddingFunction": GoogleVertexEmbeddingFunction,
        "HuggingFaceEmbeddingFunction": HuggingFaceEmbeddingFunction,
        "InstructorEmbeddingFunction": InstructorEmbeddingFunction,
        "Text2VecEmbeddingFunction": Text2VecEmbeddingFunction,
        "ONNXMiniLM_L6_V2": ONNXMiniLM_L6_V2
    }

    """
    This class acts as an adapter to connect to ChromaDB vector database.
    It extends the BaseConnection class by overidding _connect().
    It also provides other helpful methods to interact with the ChromaDB client.
    """

    @override
    def _connect(self,
                 client: str = "PersistentClient",
                 **kwargs) -> ClientAPI:

        if client == "PersistentClient":
            if "path" not in self._kwargs:
                raise Exception("`path` argument is not provided!")

            path = self._kwargs["path"]
            if not os.path.exists(path):
                raise Exception(f"Path `{path}` does not exist!")

            return chromadb.PersistentClient(
                path=path
            )

        if client == "HttpClient":
            if "host" not in self._kwargs:
                raise Exception("`host` argument is not provided!")
            if "port" not in self._kwargs:
                raise Exception("`port` argument is not provided!")

            return chromadb.HttpClient(
                host=self._kwargs["host"],
                port=self._kwargs["port"],
            )

        else:
            raise Exception("Invalid client type provided in `client` argument!")

    def create_collection(self,
                          collection_name: str,
                          embedding_function_name: str,
                          embedding_config: Dict,
                          metadata: Dict = {"hnsw:space": "l2"}) -> None:
        """
        This method creates a collection in ChromaDB that requires an embedding function and distance method.
        The embedding function is specified by the `embedding_function_name` argument.
        The `embedding_config` argument is a dictionary that contains the configuration for the embedding function.
        The `metadata` argument is a dictionary that contains the configuration for the distance method.
        """

        if embedding_function_name not in self.EMBEDDING_FUNCTION_MAP:
            raise Exception("Invalid embedding function provided in `embedding_function` argument!")

        try:
            embedding_function = self.EMBEDDING_FUNCTION_MAP[embedding_function_name](**embedding_config)
            self._instance.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata=metadata
            )
        except Exception as exception:
            raise Exception(f"Error while creating collection `{collection_name}`: {str(exception)}")


    def delete_collection(self, collection_name: str) -> None:
        """
        This method deletes a collection in ChromaDB.
        If the collection does not exist, it will raise an exception.
        """

        try:
            self._instance.delete_collection(name=collection_name)
        except Exception as exception:
            raise Exception(f"Error while deleting collection `{collection_name}`: {str(exception)}")

    def get_collection(self, collection_name: str) -> Collection:
        """
        This method gets a collection in ChromaDB.
        If the collection does not exist, it will raise an exception.
        """

        try:
            return self._instance.get_collection(name=collection_name)
        except Exception as exception:
            raise Exception(f"Error while getting collection `{collection_name}`: {str(exception)}")

    def get_all_collection_names(self) -> List:
        """
        This method gets all collection names in ChromaDB.
        """

        collection_names = []
        collections = self._instance.list_collections()
        for col in collections:
            collection_names.append(col.name)
        return collection_names

    def upload_documents(self,
                        collection_name: str,
                        documents: List,
                        metadatas: List,
                        ids: List,
                        embedding_function_name: str = "",
                        embedding_config: Dict = {},
                        embeddings: List = None) -> None:
        """
        This method uploads documents to a collection in ChromaDB.
        The `documents` argument is a list of documents, which contains list of texts to be embedded.
        The `metadatas` argument is a list of metadatas, which contains list of dictionaries that provide details about each document.
        The `embeddings` argument is a list of embeddings, which contains list of embeddings for each document.
        The `ids` argument is a list of ids, which contains list of ids for each document.

        If embeddings are not provided, the method will embed the documents using the embedding function specified in the collection.
        """        
        if embedding_function_name not in self.EMBEDDING_FUNCTION_MAP:
            raise Exception("Invalid embedding function provided in `embedding_function` argument!")

        embedding_function = cast(
            EmbeddingFunction[Documents],
            self.EMBEDDING_FUNCTION_MAP[embedding_function_name](**embedding_config),
        )

        try:
            collection = self._instance.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            for idx, doc in enumerate(documents):
                if not embeddings:
                    embedding = collection._embedding_function([doc])
                else:
                    embedding = embeddings[idx]

                collection.add(ids=ids[idx],
                                metadatas=metadatas[idx],
                                documents=doc,
                                embeddings=embedding)

        except Exception as exception:
            raise Exception(f"Error while adding document to collection `{collection_name}`: {str(exception)}")

    def update_collection_data(self,
               collection_name: str,
               ids: List,
               documents: List,
               metadatas: List,
               embedding_function_name: str = "",
               embedding_config: Dict = {},
               embeddings: List = None) -> None:
        """
        This method updates documents in a collection in ChromaDB based on their existing ids.
        """
        if embedding_function_name not in self.EMBEDDING_FUNCTION_MAP:
            raise Exception("Invalid embedding function provided in `embedding_function` argument!")

        embedding_function = cast(
            EmbeddingFunction[Documents],
            self.EMBEDDING_FUNCTION_MAP[embedding_function_name](**embedding_config),
        )

        try:
            collection = self._instance.get_collection(collection_name)
            for idx, doc in enumerate(documents):
                if not embeddings:
                    embedding = collection._embedding_function([doc])
                else:
                    embedding = embeddings[idx]

                collection.update(ids=ids[idx],
                            metadatas=metadatas[idx],
                            documents=doc,
                            embeddings=embedding)
        except Exception as exception:
            raise Exception(f"Error while updating document in collection `{collection_name}`: {str(exception)}")


    def get_collection_data(self,
                            collection_name: str,
                            attributes: List = ["documents", "embeddings", "metadatas"]):
        """
        This method gets all data from a collection in ChromaDB in form of a Pandas DataFrame.
        The `attributes` argument is a list of attributes to be included in the DataFrame.
        """

        @streamlit.cache_data(ttl=10)
        def get_data() -> pd.DataFrame:
            try:
                collection = self._instance.get_collection(collection_name)
                collection_data = collection.get(
                    include=attributes
                )
                return pd.DataFrame(data=collection_data)
            except Exception as exception:
                raise Exception(f"Error while getting data from collection `{collection_name}`: {str(exception)}")
        return get_data()

    def query(self,
              collection_name: str,
              query: List,
              where_metadata_filter: Dict = None,
              where_document_filter: Dict = None,
              num_results_limit: int = 10,
              attributes: List = ["distances", "documents", "embeddings", "metadatas", "uris", "data"],
            ) -> pd.DataFrame:
        """
        This method queries a collection in ChromaDB based on a list of query texts.
        The `where_metadata_filter` argument is a dictionary that contains the metadata filter.
        The `where_document_filter` argument is a dictionary that contains the document filter.
        The `num_results_limit` argument is the number of results to be returned.
        The `attributes` argument is a list of attributes to be included in the DataFrame.

        The return dataframe will only contain one row of result for each query text.
        """

        try:
            collection = self._instance.get_collection(collection_name)
            results = collection.query(
                query_texts=query,
                n_results=num_results_limit,
                include=attributes,
                where=where_metadata_filter,
                where_document=where_document_filter
            )
            df = pd.DataFrame()
            df["ids"] = results["ids"]

            for k in attributes:
                if results[k] is None:
                    df[k] = [None] * len(df)
                else:
                    df[k] = results[k]

            return df

        except Exception as exception:
            raise Exception(f"Error while querying collection `{collection_name}`: {str(exception)}")

import os
from streamlit.connections import BaseConnection
import chromadb
from typing_extensions import override


class ChromadbConnection(BaseConnection):

    """
    This class acts as an adapter to connect to ChromaDB vector database.
    It extends the BaseConnection class by overidding _connect().
    It also provides other helpful methods to interact with the ChromaDB client.
    """

    @override
    def _connect(self,
                 client: str = "PersistentClient",
                 **kwargs) -> chromadb.Client:

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


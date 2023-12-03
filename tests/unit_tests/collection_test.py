import os
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock
from src.streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
import streamlit


class TestCollection(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCollection, self).__init__(*args, **kwargs)

    def test_create_collection(self):
        mock_persistent_path = f"{os.getcwd()}/tests/unit_tests/create_collection_persistent"
        os.mkdir(mock_persistent_path)
        mock_connection = streamlit.connection(
            name="test_create_collection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_path
        )

        try:
            mock_connection.create_collection(
                collection_name="test_create_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )
        except Exception as ex:
            self.fail(f"create_collection() raised Exception: {str(ex)}!")
        finally:
            shutil.rmtree(mock_persistent_path)

    def test_create_collection_invalid_embedding_function(self):
        mock_persistent_path = f"{os.getcwd()}/tests/unit_tests/create_collection_persistent"
        os.mkdir(mock_persistent_path)
        mock_connection = streamlit.connection(
            name="test_create_collection_invalid_embedding_function",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_path
        )

        with self.assertRaises(Exception) as context:
            mock_connection.create_collection(
                collection_name="test_create_invalid_embedding_collection",
                embedding_function_name="InvalidEmbeddingFunction",
                embedding_config={},
            )
            self.assertTrue("Invalid embedding function provided in `embedding_function` argument!" in str(context.exception))
        shutil.rmtree(mock_persistent_path)

    def test_create_collection_existing_collection(self):
        mock_persistent_path = f"{os.getcwd()}/tests/unit_tests/create_collection_persistent"
        os.mkdir(mock_persistent_path)
        mock_connection = streamlit.connection(
            name="test_create_collection_existing_collection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_path
        )

        with self.assertRaises(Exception) as context:
            mock_connection.create_collection(
                collection_name="test_create_existing_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )

            mock_connection.create_collection(
                collection_name="test_create_existing_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )

            self.assertTrue(f"Error while creating collection `{self.collection_name}`: Collection already exists!" in str(context.exception))
        shutil.rmtree(mock_persistent_path)

    def test_delete_collection(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/delete_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_delete_collection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )

        try:
            mock_connection.create_collection(
                collection_name="test_delete_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )
            mock_connection.delete_collection(
                collection_name="test_delete_collection"
            )
        except Exception as ex:
            self.fail(f"delete_collection() raised Exception: {str(ex)}!")
        finally:
            shutil.rmtree(mock_persistent_dir)

    def test_delete_non_existing_collection(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/delete_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_delete_non_existing_collection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )

        with self.assertRaises(Exception) as context:
            mock_connection.delete_collection(
                collection_name="test_delete_non_existing_collection"
            )
            self.assertTrue(f"Error while deleting collection `{self.collection_name}`: Collection does not exist!" in str(context.exception))
        shutil.rmtree(mock_persistent_dir)


    def test_get_collection(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/get_collection_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_get_collection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )

        try:
            mock_connection.create_collection(
                collection_name="test_get_existing_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )

            mock_connection.get_collection(
                collection_name="test_get_existing_collection"
            )
        except Exception as ex:
            self.fail(f"get_collection() raised Exception: {str(ex)}!")
        finally:
            shutil.rmtree(mock_persistent_dir)

    def test_get_non_existing_collection(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/get_collection_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_get_non_existing_collection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )

        with self.assertRaises(Exception) as context:
            mock_connection.get_collection(
                collection_name="test_get_non_existing_collection"
            )
            self.assertTrue(f"Error while getting collection `{self.collection_name}`: Collection does not exist!" in str(context.exception))
        shutil.rmtree(mock_persistent_dir)

    def test_get_all_collections(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/get_all_collection_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_get_all_collections",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )
        mock_connection.create_collection(
            collection_name="test_get_existing_collection_1",
            embedding_function_name="DefaultEmbeddingFunction",
            embedding_config={},
        )
        mock_connection.create_collection(
            collection_name="test_get_existing_collection_2",
            embedding_function_name="DefaultEmbeddingFunction",
            embedding_config={},
        )

        collection_names = mock_connection.get_all_collection_names()
        self.assertGreaterEqual(len(collection_names), 2)
        shutil.rmtree(mock_persistent_dir)

    def test_add_document(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/add_docs_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_add_document",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )

        try:
           mock_connection.create_collection(
                collection_name="test_add_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )

           mock_connection.upload_document(
                collection_name="test_add_collection",
                documents=["lorem ipsum", "doc2", "doc3"],
                metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                ids=["id1", "id2", "id3"],
                embeddings=None,
            )
        except Exception as ex:
            self.fail(f"add_document() raised Exception: {str(ex)}!")
        finally:
            shutil.rmtree(mock_persistent_dir)

def test_get_collection_data(self):
        mock_persistent_dir = f"{os.getcwd()}/tests/unit_tests/get_data_persistent"
        os.mkdir(mock_persistent_dir)
        mock_connection = streamlit.connection(
            name="test_add_document",
            type=ChromadbConnection,
            client="PersistentClient",
            path=mock_persistent_dir
        )

        try:
            mock_connection.create_collection(
                collection_name="test_add_collection",
                embedding_function_name="DefaultEmbeddingFunction",
                embedding_config={},
            )

            mock_connection.upload_document(
                collection_name="test_add_collection",
                documents=["lorem ipsum", "doc2", "doc3"],
                metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                ids=["id1", "id2", "id3"],
                embeddings=None,
            )
            data = mock_connection.get_collection_data(
                collection_name="test_add_collection",
                attributes=["documents", "embeddings", "metadatas"]
            )
            print(data)
            self.assertEqual(len(data), 3)
        except Exception as ex:
            self.fail(f"get_collection_data() raised Exception: {str(ex)}!")
        finally:
            shutil.rmtree(mock_persistent_dir)

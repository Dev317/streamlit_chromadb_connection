import os
from unittest import TestCase
from unittest.mock import patch, MagicMock
from src.streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
import streamlit
import chromadb
import shutil


class TestConnect(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestConnect, self).__init__(*args, **kwargs)
        self.persistent_path = f"{os.getcwd()}/tests/unit_tests/persistent"

    def setUp(self) -> None:
        if not os.path.exists(self.persistent_path):
            os.mkdir(self.persistent_path)

    def tearDown(self) -> None:
        if os.path.exists(self.persistent_path):
            shutil.rmtree(self.persistent_path)

    def test_persistent_connect(self):
        connection = streamlit.connection(
            name="persistent_connection",
            type=ChromadbConnection,
            client="PersistentClient",
            path=self.persistent_path
        )
        self.assertEqual(connection._raw_instance._server._system.settings.is_persistent, True)
        self.assertEqual(connection._raw_instance._server._system.settings.persist_directory, self.persistent_path)

    def test_persistent_connect_missing_path(self):
        with self.assertRaises(Exception) as context:
            streamlit.connection(
                name="persistent_connection",
                type=ChromadbConnection,
                client="PersistentClient",
            )
            self.assertTrue("`path` argument is not provided!" in str(context.exception))

    def test_persistent_connect_invalid_path(self):
        invalid_path = f"{os.getcwd()}/tests/unit_tests/invalid_persistent"
        with self.assertRaises(Exception) as context:
            streamlit.connection(
                name="persistent_connection",
                type=ChromadbConnection,
                client="PersistentClient",
                path=invalid_path
            )
            self.assertTrue(f"Path `{invalid_path}` does not exist!" in str(context.exception))

    @patch("src.streamlit_chromadb_connection.chromadb_connection.ChromadbConnection._connect")
    def test_http_connect(self, mock_client_object):
        mock_client_object.return_value = MagicMock(spec=chromadb.HttpClient)
        streamlit.connection(
            name="http_connection",
            type=ChromadbConnection,
            client="HttpClient",
            host="localhost",
            port=8080
        )
        mock_client_object.assert_called_once_with(client="HttpClient", host="localhost", port=8080)

    def test_http_connect_missing_host(self):
        with self.assertRaises(Exception) as context:
            streamlit.connection(
                name="http_connection",
                type=ChromadbConnection,
                client="HttpClient",
                port=8080
            )
            self.assertTrue("`host` argument is not provided!" in str(context.exception))

    def test_http_connect_missing_port(self):
        with self.assertRaises(Exception) as context:
            streamlit.connection(
                name="http_connection",
                type=ChromadbConnection,
                client="HttpClient",
                host="localhost"
            )
            self.assertTrue("`port` argument is not provided!" in str(context.exception))


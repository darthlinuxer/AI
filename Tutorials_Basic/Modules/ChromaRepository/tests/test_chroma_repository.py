import os
from dotenv import load_dotenv

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

current_dir = os.path.dirname(os.path.realpath(__file__))
database_dir = os.path.join(current_dir, "database")
database_dir_str = str(database_dir)

from ..src import *

import unittest

from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from parameterized import parameterized


class TestChromaRepository(unittest.TestCase):
    def setUp(self):
        self.chroma_repo = ChromaRepository(
            persist_directory=database_dir_str,
            collection_name="test_collection",
            embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        )

        self.chroma_repo.add(
            texts=["this is a test"],
            metadatas=[{"source": "test", "group": "group1"}],
            ids=["1"],
        )

    def tearDown(self):
        pass

    def test_01_adding_to_chroma_with_metadata(self):
        id = self.chroma_repo.add(
            texts=["this is a 2nd test"],
            metadatas=[{"source": "test", "group": "group2"}],
            ids=["2"],
        )
        self.assertIn("2", id)

    def test_01_add_to_chroma_without_metadata(self):
        id = self.chroma_repo.add(
            texts=["this is a 3rd test without metadata"],
            ids=["3"],
        )
        doc = self.chroma_repo.getall_by_ids("3")
        self.assertIn("3", id)

    def test_02_getting_all_from_chroma(self):
        docs = self.chroma_repo.get_all()
        self.assertEqual(len(docs["ids"]), 3)

    @parameterized.expand([("1", 1), (["1", "2"], 2), (["1", "2", "3"], 3)])
    def test_02_getall_by_ids(self, input_ids, expected_length):
        docs = self.chroma_repo.getall_by_ids(
            ids=input_ids, include=["embeddings", "metadatas", "documents"]
        )
        self.assertEqual(len(docs["ids"]), expected_length)
        self.assertIsNotNone(docs["embeddings"])
        self.assertIsNotNone(docs["metadatas"])
        self.assertIsNotNone(docs["documents"])

    def test_02_getall_by_metadata(self):
        docs = self.chroma_repo.getall_by_metadata(metadata_query={"source": "test"})
        self.assertEqual(len(docs["ids"]), 2)

    @parameterized.expand([("test", 3), ("2nd", 1), ("3rd", 1)])
    def test_02_getall_text_contains(self, text, expected_docs):
        docs = self.chroma_repo.getall_text_contains(text=text)
        self.assertEqual(len(docs["ids"]), expected_docs)

    def test_02_getall_text_contains_by_metadata(self):
        docs = self.chroma_repo.getall_text_contains_by_metadata(
            text="this is a", metadata_query={"source": "test"}
        )
        self.assertEqual(len(docs["ids"]), 2)

    def test_03_context_search_by_similarity_with_score(self):
        docs = self.chroma_repo.context_search_by_similarity_with_score(
            context="this is a",
            k=5,
            metadata_filter={"source": "test"},
            document_filter={
                "$not_contains": "2nd"
            },  # "$contains", "$not_contains", "$and", "$or"
        )
        self.assertEqual(len(docs), 1)

    @parameterized.expand(
        [(None, 3), ({"source": "test"}, 2), ({"group": "group1"}, 1)]
    )
    def test_03_context_search_by_similarity(self, metadata_text, expected_docs):
        docs = self.chroma_repo.context_search_by_similarity(
            context="this is a", k=5, metadata_filter=metadata_text
        )
        self.assertEqual(len(docs), expected_docs)

    @parameterized.expand(
        [(None, 3), ({"source": "test"}, 2), ({"group": "group1"}, 1)]
    )
    def test_04_context_search_with_simpleretriever_strategy(
        self, filter, expected_docs
    ):

        retriever = SimilaritySearchRetriever(
            vectorstore=self.chroma_repo.db,
            max_number_of_documents=5,
            filter_by_metadata=filter,
        )
        docs = self.chroma_repo.context_search_by_retriever_strategy(
            context="this is a", retriever=retriever
        )
        self.assertEqual(len(docs), expected_docs)

    @parameterized.expand(
        [(None, 3), ({"source": "test"}, 2), ({"group": "group1"}, 1)]
    )
    def test_04_context_search_with_multisearchetriever_strategy(
        self, metadata_text, expected_docs
    ):

        retriever = MultiSearchRetriever.from_llm(
            vectorstore=self.chroma_repo.db,
            max_number_of_documents=5,
            llm=ChatOpenAI(api_key=OPENAI_API_KEY),
            include_original_question=True,
            metadata=metadata_text,
        )
        docs = self.chroma_repo.context_search_by_retriever_strategy(
            context="this is a test", retriever=retriever
        )
        self.assertEqual(len(docs), expected_docs)

    def test_04_context_search_with_smallchunkretriever_strategy(self):
        documents = [
            Document(
                page_content="""
                I have a dog. Her name is Pity
                The Chicago Bulls dominated the NBA in the 1990s.
                The Concorde was named after the city of Chicago.
                There's no one on Earth who doesn't like dogs.
                """,
            ),
            Document(
                page_content="""
                I heard the largest hill in the Netherlands is like 8m tall only.
                When you sneeze the speed of the air can reach up to 75km/h
                """
            ),
        ]

        byte_store = InMemoryStore()
        text_splitter = CharacterTextSplitter(
            chunk_size=100, chunk_overlap=0, separator="\n", strip_whitespace=True
        )
        retriever = SmallChunksSearchRetriever(
            documents=documents,
            child_splitter=text_splitter,
            embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
            max_number_of_documents=1,
            byte_store=byte_store,
        )

        query = "what is the Concorde named after?"
        docs = self.chroma_repo.context_search_by_retriever_strategy(
            context=query, retriever=retriever
        )
        self.assertEqual(len(docs), 1)

        sub_docs = retriever.fetch_documents(query=query)
        # distance, parent_id, sub_doc
        # (0.4120287686281514, '73b02ac14af792bf48c02f61ef6291da6c8111622bc5eb62d0ac32b643b78927', 'The Concorde was named after the city of Chicago.')
        self.assertIsNotNone(sub_docs)

    def test_04_context_search_with_smallchunkretriever_strategy_with_vectorrepository(
        self,
    ):
        documents = [
            Document(
                page_content="""
                I have a dog. Her name is Pity
                The Chicago Bulls dominated the NBA in the 1990s.
                The Concorde was named after the city of Chicago.
                There's no one on Earth who doesn't like dogs.
                """,
            ),
            Document(
                page_content="""
                I heard the largest hill in the Netherlands is like 8m tall only.
                When you sneeze the speed of the air can reach up to 75km/h
                """
            ),
        ]

        added_ids = self.chroma_repo.add(documents=documents)
        self.assertIsNotNone(added_ids)
        byte_store = InMemoryStore()

        # this configuration has the effect of splitting the documents into paragraphs
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "\n\n"],
            keep_separator=False,
            chunk_size=1,
            chunk_overlap=0,
        )

        retriever = SmallChunksSearchRetriever(
            vectorRepository=self.chroma_repo,
            child_splitter=text_splitter,
            embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
            max_number_of_documents=1,
            byte_store=byte_store,
        )

        query = "what is the Concorde named after?"
        docs = self.chroma_repo.context_search_by_retriever_strategy(
            context=query, retriever=retriever
        )
        self.assertEqual(len(docs), 1)

        sub_docs = retriever.fetch_documents(query=query)
        # distance, parent_id, sub_doc
        # (0.4120287686281514, '73b02ac14af792bf48c02f61ef6291da6c8111622bc5eb62d0ac32b643b78927', 'The Concorde was named after the city of Chicago.')
        self.assertIsNotNone(sub_docs)

    def test_05_update_by_id(self):
        document = {
            "page_content": "this is a 3rd test",
            "metadata": {"source": "test_2", "group": "group_2"},
        }
        self.chroma_repo.update_by_id("1", document=document)
        docs = self.chroma_repo.getall_by_ids(ids="1")
        self.assertEqual(docs["documents"][0], "this is a 3rd test")

    def test_05_update_by_id_with_failed_validation(self):
        document = {
            "page_content": 2,
            "metadata": {"source": "test_2", "group": "group_2"},
        }
        with self.assertRaises(ValueError) as context:
            self.chroma_repo.update_by_id("1", document=document)
        self.assertEqual(str(context.exception), "page_content must be a string")

    def test_05_update_by_id_with_failed_metadata_type(self):
        document = {"page_content": "test", "metadata": "source"}
        with self.assertRaises(ValueError) as context:
            self.chroma_repo.update_by_id("1", document=document)
        self.assertEqual(str(context.exception), "metadata must be a dict")

    def test_05_update_by_id_without_metadata(self):
        document = {"page_content": "test"}
        self.chroma_repo.update_by_id("1", document=document)
        docs = self.chroma_repo.getall_by_ids(ids="1")
        self.assertEqual(docs["documents"][0], "test")

    def test_99_delete_by_ids(self):
        self.chroma_repo.delete_by_ids("1")
        docs = self.chroma_repo.get_all()
        self.assertNotIn("1", docs["ids"])

    # @unittest.skip
    def test_100_delete_all(self):
        self.chroma_repo.delete_all()
        docs = self.chroma_repo.get_all()
        self.assertEqual(len(docs["ids"]), 0)


if __name__ == "__main__":
    unittest.main()

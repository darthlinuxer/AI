import os
import unittest
import tempfile
import shutil
from dotenv import load_dotenv

# Load .env from tests directory
current_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from chroma_repository import *

from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain_text_splitters import CharacterTextSplitter


class TestChromaRepositoryIntegration(unittest.TestCase):
    
    def setUp(self):
        if not OPENAI_API_KEY:
            self.skipTest("OPENAI_API_KEY not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        self.repo = ChromaRepository(
            persist_directory=self.temp_dir,
            collection_name="test_collection",
            embedding_function=self.embedding_function
        )

    def tearDown(self):
        try:
            self.repo.delete_all()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_retrieve_texts(self):
        texts = ["This is a test document", "Another test document"]
        ids = ["doc1", "doc2"]
        metadatas = [{"source": "test"}, {"source": "test"}]
        
        result = self.repo.add(texts=texts, ids=ids, metadatas=metadatas)
        self.assertEqual(result, ids)
        
        # Test retrieval
        retrieved = self.repo.getall_by_ids(ids)
        self.assertEqual(len(retrieved["ids"]), 2)
        self.assertIn("doc1", retrieved["ids"])
        self.assertIn("doc2", retrieved["ids"])

    def test_add_and_retrieve_documents(self):
        documents = [
            Document(page_content="Test document 1", metadata={"type": "article"}),
            Document(page_content="Test document 2", metadata={"type": "blog"})
        ]
        
        result = self.repo.add(documents=documents)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_similarity_search(self):
        texts = ["Python programming language", "Java programming language", "Machine learning with Python"]
        ids = ["py1", "java1", "ml1"]
        
        self.repo.add(texts=texts, ids=ids)
        
        # Search for Python-related content
        results = self.repo.context_search_by_similarity("Python", k=2)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)

    def test_text_search_functions(self):
        texts = ["This contains important information", "This has different content"]
        ids = ["info1", "content1"]
        
        self.repo.add(texts=texts, ids=ids)
        
        # Test text contains search
        results = self.repo.getall_text_contains("important")
        self.assertIn("info1", results["ids"])

    def test_update_document(self):
        # Add initial document
        self.repo.add(texts=["Original content"], ids=["update_test"])
        
        # Update the document
        new_document = {
            "page_content": "Updated content",
            "metadata": {"updated": True}
        }
        
        self.repo.update_by_id("update_test", new_document)
        
        # Verify update
        result = self.repo.getall_by_ids("update_test")
        self.assertEqual(result["documents"][0], "Updated content")

    def test_delete_operations(self):
        texts = ["Doc to delete", "Doc to keep"]
        ids = ["delete_me", "keep_me"]
        
        self.repo.add(texts=texts, ids=ids)
        
        # Delete specific document
        self.repo.delete_by_ids(["delete_me"])
        
        result = self.repo.get_all()
        self.assertNotIn("delete_me", result["ids"])
        self.assertIn("keep_me", result["ids"])


class TestSimilaritySearchRetrieverIntegration(unittest.TestCase):
    
    def setUp(self):
        if not OPENAI_API_KEY:
            self.skipTest("OPENAI_API_KEY not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        self.repo = ChromaRepository(
            persist_directory=self.temp_dir,
            collection_name="test_collection",
            embedding_function=self.embedding_function
        )
        
        # Add test data
        texts = ["Python is a programming language", "Java is also a programming language", "Cars are vehicles"]
        ids = ["py", "java", "car"]
        self.repo.add(texts=texts, ids=ids)

    def tearDown(self):
        try:
            self.repo.delete_all()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_similarity_search_retriever(self):
        retriever = SimilaritySearchRetriever(
            vectorstore=self.repo.db,
            max_number_of_documents=2
        )
        
        results = retriever.fetch_documents("programming")
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)

    def test_similarity_search_retriever_with_filter(self):
        # Add documents with metadata
        self.repo.add(
            texts=["Python tutorial", "Java tutorial"],
            ids=["py_tut", "java_tut"],
            metadatas=[{"category": "tutorial"}, {"category": "tutorial"}]
        )
        
        retriever = SimilaritySearchRetriever(
            vectorstore=self.repo.db,
            max_number_of_documents=5,
            filter_by_metadata={"category": "tutorial"}
        )
        
        results = retriever.fetch_documents("tutorial")
        self.assertIsInstance(results, list)


class TestMultiSearchRetrieverIntegration(unittest.TestCase):
    
    def setUp(self):
        if not OPENAI_API_KEY:
            self.skipTest("OPENAI_API_KEY not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
        
        self.repo = ChromaRepository(
            persist_directory=self.temp_dir,
            collection_name="test_collection",
            embedding_function=self.embedding_function
        )
        
        # Add test data
        texts = ["Python programming basics", "Advanced Python concepts", "Java fundamentals"]
        ids = ["py1", "py2", "java1"]
        self.repo.add(texts=texts, ids=ids)

    def tearDown(self):
        try:
            self.repo.delete_all()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multi_search_retriever(self):
        retriever = MultiSearchRetriever.from_llm(
            vectorstore=self.repo.db,
            max_number_of_documents=3,
            llm=self.llm,
            include_original_question=True
        )
        
        results = retriever.fetch_documents("Python programming")
        self.assertIsInstance(results, list)


class TestSmallChunksSearchRetrieverIntegration(unittest.TestCase):
    
    def setUp(self):
        if not OPENAI_API_KEY:
            self.skipTest("OPENAI_API_KEY not available")
            
        self.embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.documents = [
            Document(page_content="Python is a powerful programming language. It's used for web development, data science, and AI."),
            Document(page_content="Java is another popular programming language. It's known for its platform independence.")
        ]

    def test_small_chunks_retriever_with_documents(self):
        byte_store = InMemoryStore()
        splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0, separator=".")
        
        retriever = SmallChunksSearchRetriever(
            documents=self.documents,
            vectorRepository=None,
            filter_database=None,
            child_splitter=splitter,
            embedding_function=self.embedding_function,
            byte_store=byte_store,
            max_number_of_documents=2
        )
        
        results = retriever.fetch_documents("programming language")
        self.assertIsInstance(results, list)
        if results:
            # Each result should be a tuple of (distance, parent_id, subdoc_content)
            self.assertIsInstance(results[0], tuple)
            self.assertEqual(len(results[0]), 3)


if __name__ == "__main__":
    unittest.main()
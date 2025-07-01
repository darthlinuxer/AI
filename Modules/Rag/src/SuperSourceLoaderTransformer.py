from typing import Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import semchunk
import tiktoken


class SuperSourceLoaderTransformer:

    def loadDataFromSource(self, source: str, **kwargs) -> list[Document]:
        """
        A method to load data from a source if the source is an http link.
        It checks if the source is a youtube URL and loads the transcript if it is.
        It also handles loading from PDF, DOC, DOCX, and PDF files, returning a list of Documents.
        """
        # create a code to load data from source if source is http

        def _is_youtube_url(url: str) -> bool:
            import re

            pattern = r"^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$"
            return re.match(pattern, url) is not None

        if _is_youtube_url(source):
            from langchain_community.document_loaders import YoutubeLoader

            try:
                language = "en"
                # check if there is a key named language in kwargs
                if "language" in kwargs:
                    language = kwargs["language"]

                loader = YoutubeLoader.from_youtube_url(
                    source,
                    add_video_info=True,
                    language=[language],
                    # translation="pt"
                )
                transcript = loader.load()
                return transcript
            except Exception as e:
                print(f"Error when loading data from youtube URL: {source}. {e}")
                return []

        if source.startswith("http"):
            if "pdf" in source:
                from langchain_community.document_loaders import OnlinePDFLoader

                loader_onlinepdf = OnlinePDFLoader(source)
                return loader_onlinepdf.load()

            from langchain_community.document_loaders import WebBaseLoader

            loader_HTML = WebBaseLoader(web_paths=[source])
            docsWeb = loader_HTML.load()
            return docsWeb

        if any(ext in source for ext in [".doc", ".docx"]):
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(source)
            data = loader.load()
            return data

        if ".pdf" in source:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(source)
            data = loader.load()
            return data

    def create_hashids_for_texts(self, texts: list[str]) -> list[str]:
        import hashlib

        return [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    def create_hashids_for_documents(self, documents: list[Document]) -> list[str]:
        import hashlib

        return [
            hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            for doc in documents
        ]

    def semantic_chunk_documents(
        self,
        documents: list[Document],
        embed_model: OpenAIEmbeddings | OllamaEmbeddings,
        threshold: int = 60,
    ) -> list[Document]:
        """
        Splits a list of Document objects into semantic chunks based on a specified threshold.

        Args:
            documents (list[Document]): A list of Document objects to be split into semantic chunks.
            threshold (int, optional): The threshold value for determining the breakpoints of semantic chunks. Defaults to 60.

        Returns:
            list[Document]: A list of Document objects representing the semantic chunks.
        """

        text_splitter = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold,
        )
        return text_splitter.split_documents(documents)

    def semantic_chunk_text(
        self,
        text: str,
        embed_model: OpenAIEmbeddings | OllamaEmbeddings,
        threshold: int = 60,
    ) -> list[str]:
        """
        Splits a given text into semantic chunks based on a specified threshold.

        Args:
            text (str): The input text to be split into semantic chunks.
            threshold (int, optional): The threshold value for determining the breakpoints of semantic chunks. Defaults to 60.

        Returns:
            list[Document]: A list of Document objects representing the semantic chunks.
        """

        text_splitter = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold,
        )
        return text_splitter.split_text(text)

    def semantic_chunk_with_token_size(self, text: str, chunk_size: int) -> list[str]:
        encoder = tiktoken.get_encoding("cl100k_base")
        token_counter = lambda text: len(encoder.encode(text))
        return semchunk.chunk(
            text, chunk_size=chunk_size, token_counter=token_counter, _recursion_depth=4
        )

    def split_by_separators(
        self, input_data: Any, separators=["\n\n"]
    ) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators, chunk_size=1, chunk_overlap=0
        )
        if isinstance(input_data, list) and all(
            isinstance(doc, Document) for doc in input_data
        ):
            return text_splitter.split_documents(input_data)
        elif isinstance(input_data, str):
            return text_splitter.split_text(input_data)
        else:
            raise ValueError(
                "Unsupported input type, it has to be a string or a list of Document"
            )

    def ai_semantic_chunk(
        self, text: str, llm_model: ChatOpenAI | ChatOllama
    ) -> list[str]:
        from langchain_core.prompts import ChatPromptTemplate

        prompt_template = """
        Given the following context:
        ===========================
        {text}
        ===========================
        Instructions: 
        1. Return a list of strings with the given context separated by semantic similar sentences
        2. Do not invent new words
        3. Do not summarize the context
        4. Do not return topics , only the semantic similar sentences
        5. Do not return any other text other than the semantic similar sentences
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return (
            {"text": RunnablePassthrough()} | prompt | llm_model | StrOutputParser()
        ).invoke(text)

    def calculate_tokens(self, chunks: list[str]) -> list[int]:
        encoder = tiktoken.get_encoding("cl100k_base")
        token_counter = lambda text: len(encoder.encode(text))
        return [token_counter(chunk) for chunk in chunks]

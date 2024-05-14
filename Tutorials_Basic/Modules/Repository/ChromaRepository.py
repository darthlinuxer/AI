from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from chromadb.types import Where
from .IVectorRepository import IVectorRepository

class ChromaRepository(IVectorRepository):

    __DEFAULT_INCLUDES = ["metadatas", "documents"]
    __ALL_INCLUDES = ["metadatas", "documents", "embeddings"]

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_function: Embeddings,
    ) -> None:

        self._db = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

    @property
    def db(self) -> Chroma:
        return self._db

    def get_vectordb(self) -> VectorStore:
        return self.db
    
    def get_all(self, limit=10, offset=0, **kwargs) -> Dict[str, Any]:
        """
        Retrieves all documents from the database.

        Args:
            limit (int, optional): The maximum number of documents to retrieve. Defaults to 10.
            offset (int, optional): The number of documents to skip before starting to retrieve. Defaults to 0.
            **kwargs: Additional keyword arguments.
            - include (List[str]): The fields to include in the retrieved documents. Defaults to self._DEFAULT_INCLUDES.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved documents. The keys are the document IDs, and the values are the documents themselves.

        """
        return self.db.get(
            limit=limit,
            offset=offset,
            include=kwargs.get("include", self.__DEFAULT_INCLUDES),
        )

    def getall_by_ids(self, ids: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Retrieves one or more documents from the database by their IDs.

        Args:
            ids (Union[str, List[str]]): The ID(s) of the document(s) to retrieve.
            **kwargs: Additional keyword arguments.
                - include (List[str]): The fields to include in the retrieved documents. Defaults to self._DEFAULT_INCLUDES.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved documents. The keys are the document IDs, and the values are the documents themselves.
        """
        return self.db.get(
            ids=ids, include=kwargs.get("include", self.__DEFAULT_INCLUDES)
        )

    def getall_text_contains(
        self, text: str, limit=10, offset=0, **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieves documents containing the specified text.

        Args:
            text (str): The text to search for within the documents.
            limit (int): The maximum number of documents to retrieve. Defaults to 10.
            offset (int): The number of documents to skip before starting to retrieve. Defaults to 0.
            **kwargs: Additional keyword arguments.
                - include (List[str]): The fields to include in the retrieved documents. Defaults to self._DEFAULT_INCLUDES.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved documents. The keys are the document IDs, and the values are the documents themselves.
        """
        return self.db.get(
            where_document={"$contains": text},
            limit=limit,
            offset=offset,
            include=kwargs.get("include", self.__DEFAULT_INCLUDES),
        )

    def getall_text_contains_by_metadata(
        self, text: str, metadata_query: Where, limit=10, offset=0, **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieves documents containing the specified text, filtered by the given metadata query.

        Args:
            text (str): The text to search for within the documents.
            metadata_query (Where): The query to filter the documents by metadata.
            limit (int, optional): The maximum number of documents to retrieve. Defaults to 10.
            offset (int, optional): The number of documents to skip before starting to retrieve. Defaults to 0.
            **kwargs: Additional keyword arguments.
                - include (List[str]): The fields to include in the retrieved documents. Defaults to self._DEFAULT_INCLUDES.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved documents. The keys are the document IDs, and the values are the documents themselves.
        """
        return self.db.get(
            where_document={"$contains": text},
            where=metadata_query,
            limit=limit,
            offset=offset,
            include=kwargs.get("include", self.__DEFAULT_INCLUDES),
        )

    def getall_by_metadata(
        self, metadata_query: Where, limit=10, offset=0, **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieves all documents from the database that match the given metadata query.

        Args:
            metadata_query (Where): The query to filter the documents by metadata.
            limit (int, optional): The maximum number of documents to retrieve. Defaults to 10.
            offset (int, optional): The number of documents to skip before starting to retrieve. Defaults to 0.
            **kwargs: Additional keyword arguments.
                - include (List[str]): The fields to include in the retrieved documents. Defaults to self._DEFAULT_INCLUDES.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved documents. The keys are the document IDs, and the values are the documents themselves.
        """
        return self.db.get(
            where=metadata_query,
            limit=limit,
            offset=offset,
            include=kwargs.get("include", self.__DEFAULT_INCLUDES),
        )

    def context_search_by_similarity_with_score(
        self,
        context: str,
        k=5,
        metadata_filter: Optional[Dict[str, str]] = None,
        document_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        return self.db.similarity_search_with_score(
            context, k=k, filter=metadata_filter, where_document=document_filter
        )

    def context_search_by_similarity(
        self,
        context: str,
        k=5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        return self.db.similarity_search(context, k=k, filter=metadata_filter)

    def context_search_by_retriever_strategy(
        self,
        context: str,
        retriever: VectorStoreRetriever
    ) -> List[Document]:
        return retriever.invoke(
            input=context
        )

    def add(self, **kwargs) -> List[str]:
        """
        Adds documents or texts with their corresponding metadatas and ids to the database.

        Parameters:
            **kwargs (dict): A dictionary containing the following keys:
                - documents (list): A list of Document objects to be added to the database.
                - texts (list): A list of strings representing the texts to be added to the database.
                - metadatas (list): A list of dictionaries representing the metadatas of the texts.
                - ids (list): A list of strings representing the ids of the texts.

        Returns:
            - If documents is provided: A list of strings representing the ids of the added documents.
            - If texts, metadatas, and ids are provided: A list of strings representing the ids of the added texts.

        Raises:
            - ValueError: If neither documents nor texts, metadatas, and ids are provided.
        """
        try:
            if "documents" in kwargs:
                return self.db.add_documents(kwargs["documents"])
            elif "texts" in kwargs and "metadatas" in kwargs and "ids" in kwargs:
                return self.db.add_texts(
                    texts=kwargs["texts"], metadatas=kwargs["metadatas"], ids=kwargs["ids"]
                )
            elif "texts" in kwargs and "ids" in kwargs:
                return self.db.add_texts(
                    texts=kwargs["texts"], metadatas=[], ids=kwargs["ids"]
                )
            else:
                raise ValueError(
                    "Must provide either documents or texts+ids or texts+ids+metadatas"
                )
        except Exception as e:
            print(f"An error has occured: {e}") 
            raise

    def delete_by_ids(self, ids: List[str]) -> None:
        self.db.delete(ids=ids)

    def delete_all(self) -> None:
        docs = self.get_all(limit=1000000)
        self.delete_by_ids(docs["ids"])

    def update_by_id(self, id: str, **kwargs) -> None:
        """
        A function to update a document by its id.

        Parameters:
            - id (str): The id of the document to update.
            - **kwargs: Additional keyword arguments {'document':dict[str,str]}

        Raises:
            - ValueError: If the kwargs do not contain the 'document' key.

        Returns:
            - None
        """
        if "document" not in kwargs:
            raise ValueError("Must provide document in kwargs")

        document = kwargs["document"]

        def validate_document(document):
            if not isinstance(document, dict):
                raise ValueError("document must be a dict")
            if "page_content" not in document:
                raise ValueError(
                    "document must contain page_content and optionally a metadata"
                )
            if not isinstance(document.get("page_content"), str):
                raise ValueError("page_content must be a string")
            if "metadata" in document:
                if not isinstance(document.get("metadata"), dict):
                    raise ValueError("metadata must be a dict")

        validate_document(document)
        if "metadata" in document:
            document = Document(
                page_content=document["page_content"], metadata=document["metadata"]
            )
        else:
            document = Document(
                page_content=document["page_content"], metadata={"description": ""}
            )
        self.db.update_document(document_id=id, document=document)

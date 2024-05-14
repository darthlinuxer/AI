from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore

class IVectorRepository(ABC):
    
    @abstractmethod
    def get_vectordb(self) -> VectorStore:
        pass
    @abstractmethod
    def get_all(self, limit=10, offset=0, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def getall_by_ids(self, ids: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def getall_text_contains(
        self, text: str, limit=10, offset=0, **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def getall_text_contains_by_metadata(
        self, text: str, metadata_query: Dict, limit=10, offset=0, **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def getall_by_metadata(
        self, metadata_query: Dict, limit=10, offset=0, **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def context_search_by_similarity_with_score(
        self,
        query: str,
        k=5,
        metadata_filter: Dict[str, str] = None,
        document_filter: Dict[str, str] = None,
    ) -> List[Tuple[Any, float]]:
        pass

    @abstractmethod
    def context_search_by_retriever_strategy(
        self, context: str, retriever: VectorStoreRetriever
    ) -> List[Document]:
        pass

    @abstractmethod
    def add(self, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    def delete_all(self) -> None:
        pass

    @abstractmethod
    def update_by_id(self, id: str, **kwargs) -> None:
        pass

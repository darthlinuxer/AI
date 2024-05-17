from abc import ABC, abstractmethod
from typing import Any, List


class ICustomRetriever(ABC):
    @abstractmethod
    def fetch_documents(self, query: str) -> List[Any]:
        pass

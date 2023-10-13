from abc import ABC, abstractmethod
from typing import List
from web_api.short_term_memory import ShortTermMemoryImpl
from web_api.long_term_memory import LongTermMemoryImpl
from web_api.working_memory import WorkingMemoryImpl 


class Memory(ABC):
    @abstractmethod
    def retrieve_examples_from_knowledge_base(self):
        pass

class MemoryImpl(Memory): 
    def __init__(self, similarity_model_path: str, fine_tune_similarity_model: bool = False, document_db_file_path: str = None, vector_store_file_path: str = None):
        # initialize ShortTermMemory, LongTermMemory, WorkingMemory, LLM
        self.short_term_memory = ShortTermMemoryImpl()
        self.long_term_memory = LongTermMemoryImpl(similarity_model_path=similarity_model_path, fine_tune_similarity_model=fine_tune_similarity_model, document_db_file_path=document_db_file_path, vector_store_file_path=vector_store_file_path)
        self.working_memory = WorkingMemoryImpl()

    def retrieve_examples_from_knowledge_base(self, query: str, k: int = 3):
        # retrieve examples from knowledge base
        top_k_examples = self.long_term_memory.retrieve_examples_from_knowledge_base(query=query)
        return top_k_examples

    def retrieve_documents_from_knowledge_base(self, query:str, k: int = 3):
        # retrieve documents from knowledge base
        top_k_documents = self.long_term_memory.retrieve_documents_from_knowledge_base(query=query)
        return top_k_documents 

    def get_document_from_knowledge_base(self, document_id: int):
        # ask long term memory for document
        document = self.long_term_memory.get_document_from_knowledge_base(document_id=document_id)
        return document

    def get_document_type_from_knowledge_base(self, document_id: int):
        document_type = self.long_term_memory.get_document_type_from_knowledge_base(document_id=document_id)
        return document_type

    def get_current_state(self):
        current_state = self.short_term_memory.get_current_state()
        return current_state
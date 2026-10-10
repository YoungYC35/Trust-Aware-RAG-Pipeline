"""
LlamaIndex RAG Engine Module.

Manages in-memory VectorStoreIndex, Document abstractions, and raw retrieval.
Guarantees zero ground-truth label leakage into vector embeddings.
"""

from typing import List, Dict, Any, Optional

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM


class LlamaIndexRAG:
    """Manages the in-memory VectorStoreIndex, retriever (similarity_top_k=3), and optional LLM generator."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator: Optional[Any] = None,
    ):
        print(f"[Module 1] Initializing LlamaIndex RAG Engine with embedding: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = self.embed_model
        Settings.llm = MockLLM()
        self.index: Optional[VectorStoreIndex] = None
        self.retriever = None
        self.generator = generator

    def set_generator(self, generator: Any) -> None:
        """Sets or replaces the active LLM generator."""
        self.generator = generator

    def generate_answer(self, query: str, nodes: List[NodeWithScore]) -> str:
        """Generates an answer using the attached LLM generator."""
        if self.generator is None:
            return "[No generator configured]"
        return self.generator.generate_rag_answer(query, nodes)

    @property
    def sentence_transformer(self) -> Any:
        """Returns the underlying SentenceTransformer instance to avoid redundant model loading."""
        return self.embed_model._model

    def build_index(self, documents_data: List[Dict[str, Any]]) -> None:
        """Constructs an in-memory VectorStoreIndex from document dictionaries."""
        docs = []
        for item in documents_data:
            doc = Document(
                text=item["text"],
                doc_id=item["doc_id"],
                metadata={
                    "doc_id": item["doc_id"],
                    "is_injected": item["is_injected"],
                    "probe_type": item["probe_type"],
                    "split": item.get("split", "unknown"),
                    "domain": item.get("domain", "general"),
                    "attack_family": item.get("attack_family", "unknown"),
                    "reference_answer": item.get("reference_answer", ""),
                    "observable_marker": item.get("observable_marker", ""),
                },
                excluded_embed_metadata_keys=[
                    "doc_id", "is_injected", "probe_type", "split", "domain",
                    "attack_family", "reference_answer", "observable_marker"
                ],
                excluded_llm_metadata_keys=[
                    "doc_id", "is_injected", "probe_type", "split", "domain",
                    "attack_family", "reference_answer", "observable_marker"
                ],
            )
            docs.append(doc)

        self.index = VectorStoreIndex.from_documents(docs)
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        print(f"[Module 1] In-memory VectorStoreIndex constructed with {len(docs)} documents.")

    def retrieve_raw(self, query: str) -> List[NodeWithScore]:
        """Baseline retrieval without safety post-processing."""
        if self.retriever is None:
            raise ValueError("Index has not been built yet.")
        return self.retriever.retrieve(query)

"""
Embedding-based semantic search for Samsung product and content catalogs.
Uses sentence-transformers + FAISS for fast approximate nearest neighbor retrieval.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexStats:
    total_docs: int
    embedding_dim: int
    index_type: str


class EmbeddingIndexer:
    """
    Builds and queries a FAISS flat index over sentence embeddings.
    Designed for Samsung catalog search: products, media, knowledge base articles.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            self.model = SentenceTransformer(model_name)
            self._faiss = faiss
        except ImportError:
            self.model = None
            self._faiss = None

        self.dim: Optional[int] = None
        self.index = None
        self.doc_store: List[Tuple[str, str, dict]] = []  # (doc_id, text, metadata)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            # Fallback for environments without sentence-transformers
            return np.random.randn(len(texts), 384).astype(np.float32)
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def build(self, documents: List[dict], text_field: str = "text", id_field: str = "id"):
        """
        documents: list of dicts with at least `text_field` and `id_field`.
        Extra keys are stored as metadata.
        """
        texts = [d[text_field] for d in documents]
        vecs = self._embed(texts)
        self.dim = vecs.shape[1]

        if self._faiss:
            self.index = self._faiss.IndexFlatIP(self.dim)  # inner product = cosine (normalized)
            self.index.add(vecs)
        else:
            # Pure numpy fallback
            self.index = vecs

        self.doc_store = [
            (str(d[id_field]), d[text_field], {k: v for k, v in d.items() if k not in (text_field, id_field)})
            for d in documents
        ]

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        q_vec = self._embed([query])  # (1, dim)

        if self._faiss and self.index is not None and not isinstance(self.index, np.ndarray):
            scores, indices = self.index.search(q_vec, top_k)
            scores, indices = scores[0], indices[0]
        else:
            # numpy cosine similarity
            mat = self.index if isinstance(self.index, np.ndarray) else np.zeros((0, 1))
            sims = (mat @ q_vec.T).squeeze()  # (n,)
            indices = np.argsort(-sims)[:top_k]
            scores = sims[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.doc_store):
                continue
            doc_id, text, meta = self.doc_store[idx]
            results.append(SearchResult(doc_id=doc_id, text=text, score=float(score), metadata=meta))
        return results

    def stats(self) -> IndexStats:
        return IndexStats(
            total_docs=len(self.doc_store),
            embedding_dim=self.dim or 0,
            index_type="faiss.IndexFlatIP" if self._faiss else "numpy",
        )

    def save(self, path: str):
        import pickle
        import os

        os.makedirs(path, exist_ok=True)
        if self._faiss and self.index is not None and not isinstance(self.index, np.ndarray):
            self._faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        else:
            np.save(os.path.join(path, "index.npy"), self.index)
        with open(os.path.join(path, "docstore.pkl"), "wb") as f:
            pickle.dump(self.doc_store, f)

    def load(self, path: str):
        import pickle
        import os

        faiss_path = os.path.join(path, "index.faiss")
        npy_path = os.path.join(path, "index.npy")
        if self._faiss and os.path.exists(faiss_path):
            self.index = self._faiss.read_index(faiss_path)
            self.dim = self.index.d
        elif os.path.exists(npy_path):
            self.index = np.load(npy_path)
            self.dim = self.index.shape[1]
        with open(os.path.join(path, "docstore.pkl"), "rb") as f:
            self.doc_store = pickle.load(f)

"""
On-device NLP pipeline for Samsung Tizen/Android embedded environments.
Lightweight tokenization, embedding, and classification under strict resource constraints.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import json
import os


@dataclass
class InferenceConfig:
    max_seq_len: int = 64
    embedding_dim: int = 128
    quantize: bool = True
    memory_budget_mb: float = 32.0


@dataclass
class NLPOutput:
    tokens: List[str]
    embedding: List[float]
    label: str
    confidence: float
    latency_ms: float


class Vocab:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.pad_id = self.token2id.get("[PAD]", 0)
        self.unk_id = self.token2id.get("[UNK]", 1)

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = text.lower().split()
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids[:max_len]


class EmbeddingTable:
    """Static embedding table loaded from pre-quantized numpy array."""

    def __init__(self, weight_path: str, dim: int = 128):
        self.dim = dim
        if os.path.exists(weight_path):
            self.weights = np.load(weight_path).astype(np.float32)
        else:
            # Placeholder for unit tests without actual weights
            self.weights = np.random.randn(30000, dim).astype(np.float32)

    def lookup(self, ids: List[int]) -> np.ndarray:
        vecs = self.weights[ids]  # (seq_len, dim)
        return vecs.mean(axis=0)  # mean pooling


class OnDeviceNLPPipeline:
    """
    Minimal NLP pipeline designed for Samsung on-device inference.
    Avoids heavy framework dependencies; uses only numpy for inference.
    """

    def __init__(
        self,
        vocab_path: str,
        embedding_path: str,
        classifier_path: str,
        labels: List[str],
        config: Optional[InferenceConfig] = None,
    ):
        self.config = config or InferenceConfig()
        self.vocab = Vocab(vocab_path)
        self.embeddings = EmbeddingTable(embedding_path, self.config.embedding_dim)
        self.labels = labels
        # Classifier: simple linear layer stored as numpy array
        if os.path.exists(classifier_path):
            data = np.load(classifier_path)
            self.W = data["W"].astype(np.float32)  # (num_classes, dim)
            self.b = data["b"].astype(np.float32)  # (num_classes,)
        else:
            n = len(labels)
            self.W = np.random.randn(n, self.config.embedding_dim).astype(np.float32)
            self.b = np.zeros(n, dtype=np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def run(self, text: str) -> NLPOutput:
        import time

        start = time.perf_counter()
        ids = self.vocab.encode(text, self.config.max_seq_len)
        emb = self.embeddings.lookup(ids)
        logits = self.W @ emb + self.b
        probs = self._softmax(logits)
        label_idx = int(probs.argmax())
        elapsed_ms = (time.perf_counter() - start) * 1000

        return NLPOutput(
            tokens=text.lower().split()[: self.config.max_seq_len],
            embedding=emb.tolist(),
            label=self.labels[label_idx],
            confidence=round(float(probs[label_idx]), 4),
            latency_ms=round(elapsed_ms, 2),
        )

    def batch_run(self, texts: List[str]) -> List[NLPOutput]:
        return [self.run(t) for t in texts]

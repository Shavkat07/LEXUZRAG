"""
lexuz.indexer.embedder

Модуль векторизации текста с использованием многоязычной модели BAAI/bge-m3.
Генерирует:
1. Плотный вектор (Dense 1024-d, L2/Cosine)
2. Разреженный вектор (Sparse lexical weights, аналог точного BM25)
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Enable expandable segments for PyTorch CUDA allocator to prevent VRAM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from qdrant_client import models

logger = logging.getLogger("lexuz.indexer.embedder")

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DENSE_DIM = 1024
DEFAULT_MAX_LENGTH = 1024
DEFAULT_ENCODE_BATCH_SIZE = 32


class BGEM3Embedder:
    """
    Векторизатор на базе BAAI/bge-m3.
    Поддерживает одновременную генерацию Dense и Sparse представлений.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self._model = None

    def clear_cache(self) -> None:
        """Очищает кэш CUDA памяти при наличии доступного GPU."""
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        """Ленивая инициализация модели."""
        if self._model is None:
            logger.info(
                "Loading BGEM3FlagModel '%s' (device=%s, fp16=%s)...",
                self.model_name,
                self.device,
                self.use_fp16,
            )
            try:
                from FlagEmbedding import BGEM3FlagModel
                # FlagEmbedding handles cuda / cpu
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.use_fp16 if self.device == "cuda" else False,
                    device=self.device,
                )
            except ImportError as e:
                logger.error(
                    "FlagEmbedding is not installed. Please install it with 'pip install FlagEmbedding'."
                )
                raise ImportError("FlagEmbedding library required for BGEM3Embedder") from e
        return self._model

    @staticmethod
    def format_sparse_vector(lexical_dict: Dict[Union[str, int], float]) -> models.SparseVector:
        """Преобразует словарь лексических весов FlagEmbedding в SparseVector Qdrant."""
        if not lexical_dict:
            return models.SparseVector(indices=[], values=[])

        indices: List[int] = []
        values: List[float] = []
        for k, v in lexical_dict.items():
            indices.append(int(k))
            values.append(float(v))

        return models.SparseVector(indices=indices, values=values)

    def encode_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Векторизует список нормативных текстов.
        Возвращает:
        {
            "dense": List[List[float]],
            "sparse": List[models.SparseVector]
        }
        """
        if not texts:
            return {"dense": [], "sparse": []}

        enc_batch_size = batch_size if batch_size is not None else self.batch_size

        try:
            output = self.model.encode(
                texts,
                batch_size=enc_batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as e:
            # Check for CUDA OOM and attempt recovery with smaller batch size
            err_msg = str(e).lower()
            if "out of memory" in err_msg and enc_batch_size > 1:
                new_bs = max(1, enc_batch_size // 2)
                logger.warning(
                    "CUDA OOM encountered during encode (batch_size=%d). "
                    "Clearing cache and adapting batch_size to %d...",
                    enc_batch_size,
                    new_bs,
                )
                self.batch_size = new_bs
                self.clear_cache()
                return self.encode_documents(texts, batch_size=new_bs)
            raise
        finally:
            self.clear_cache()

        dense_vecs = output["dense_vecs"]
        if hasattr(dense_vecs, "tolist"):
            dense_list = dense_vecs.tolist()
        else:
            dense_list = [list(map(float, v)) for v in dense_vecs]

        lexical_weights = output["lexical_weights"]
        sparse_list = [self.format_sparse_vector(item) for item in lexical_weights]

        return {
            "dense": dense_list,
            "sparse": sparse_list,
        }

    def encode_queries(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Векторизует поисковые запросы.
        """
        if not queries:
            return {"dense": [], "sparse": []}

        enc_batch_size = (
            batch_size
            if batch_size is not None
            else min(len(queries), self.batch_size)
        )

        try:
            output = self.model.encode(
                queries,
                batch_size=enc_batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg and enc_batch_size > 1:
                new_bs = max(1, enc_batch_size // 2)
                logger.warning(
                    "CUDA OOM encountered during query encode (batch_size=%d). "
                    "Clearing cache and adapting batch_size to %d...",
                    enc_batch_size,
                    new_bs,
                )
                self.batch_size = new_bs
                self.clear_cache()
                return self.encode_queries(queries, batch_size=new_bs)
            raise
        finally:
            self.clear_cache()

        dense_vecs = output["dense_vecs"]
        if hasattr(dense_vecs, "tolist"):
            dense_list = dense_vecs.tolist()
        else:
            dense_list = [list(map(float, v)) for v in dense_vecs]

        lexical_weights = output["lexical_weights"]
        sparse_list = [self.format_sparse_vector(item) for item in lexical_weights]

        return {
            "dense": dense_list,
            "sparse": sparse_list,
        }


class MockEmbedder:
    """
    Мок-векторизатор для юнит-тестов и CI/CD окружений без скачивания весов.
    Генерирует детерминированные псевдо-векторы правильной размерности.
    """

    def __init__(self, dense_dim: int = DEFAULT_DENSE_DIM):
        self.dense_dim = dense_dim

    def clear_cache(self) -> None:
        """Мок для совместимости интерфейса."""
        pass

    def _pseudo_vector(self, text: str) -> List[float]:
        # Детерминированный хеш
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dense_dim):
            byte_val = h[i % len(h)]
            val = (byte_val - 128) / 128.0
            vec.append(round(val, 4))
        # Простейшая нормализация
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in vec]

    def _pseudo_sparse(self, text: str) -> models.SparseVector:
        tokens = text.lower().split()
        indices = []
        values = []
        seen = set()
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:6], 16) % 100000
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
                values.append(round(1.0 + (len(tok) % 5) * 0.2, 3))
        return models.SparseVector(indices=indices, values=values)

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None) -> Dict[str, Any]:
        return {
            "dense": [self._pseudo_vector(t) for t in texts],
            "sparse": [self._pseudo_sparse(t) for t in texts],
        }

    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None) -> Dict[str, Any]:
        return {
            "dense": [self._pseudo_vector(q) for q in queries],
            "sparse": [self._pseudo_sparse(q) for q in queries],
        }

"""
lexuz.storage.qdrant_manager

Модуль управления хранилищем векторов Qdrant:
- Подключение (REST + gRPC)
- Healthcheck
- Создание и конфигурация гибридной коллекции (Dense Cosine + Sparse IDF)
- Скалярное int8-квантование для оптимизации памяти
- Индексация payload-полей (lang, doc_id, unit_type, okoz_codes, archetype)
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger("lexuz.storage.qdrant_manager")

DEFAULT_COLLECTION_NAME = "lexuz_collection"
DEFAULT_DENSE_DIM = 1024
DEFAULT_INDEXED_FIELDS = [
    "lang",
    "doc_id",
    "unit_type",
    "okoz_codes",
    "archetype",
]


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Генерирует детерминированный UUID v5 на основе chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantManager:
    """Менеджер коллекции Qdrant для корпуса Lex.uz."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[QdrantClient] = None,
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc

        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                api_key=api_key,
                timeout=timeout,
            )

    def healthcheck(self) -> bool:
        """Проверяет доступность сервера Qdrant."""
        try:
            # Получение коллекций служит надежным healthcheck
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Healthcheck failed for Qdrant at %s:%d: %s", self.host, self.port, e)
            return False

    def collection_exists(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> bool:
        """Проверяет, существует ли коллекция с заданным именем."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error("Error checking collection existence: %s", e)
            raise

    def create_collection(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        recreate: bool = False,
        dense_dim: int = DEFAULT_DENSE_DIM,
    ) -> bool:
        """
        Создает коллекцию с гибридными векторами (dense + sparse)
        и int8-скалярным квантованием.
        
        Если recreate=True и коллекция существует, она будет удалена и пересоздана.
        Если recreate=False и коллекция уже существует, возвращает False.
        """
        exists = self.collection_exists(collection_name)
        if exists:
            if not recreate:
                logger.info("Collection '%s' already exists. Skipping creation.", collection_name)
                return False
            logger.info("Deleting existing collection '%s' before recreating...", collection_name)
            self.client.delete_collection(collection_name=collection_name)

        logger.info(
            "Creating collection '%s' (dense_dim=%d, sparse=IDF, quantization=int8)...",
            collection_name,
            dense_dim,
        )

        vectors_config = {
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            )
        }

        sparse_vectors_config = {
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(
                    on_disk=False,
                ),
            )
        }

        quantization_config = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            )
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            quantization_config=quantization_config,
        )

        self.create_payload_indexes(collection_name=collection_name)
        logger.info("Collection '%s' created successfully with payload indexes.", collection_name)
        return True

    def create_payload_indexes(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        fields: Optional[List[str]] = None,
    ) -> None:
        """Создает keyword индексы для фильтрации по полям payload."""
        if fields is None:
            fields = DEFAULT_INDEXED_FIELDS

        for field_name in fields:
            logger.info("Creating keyword index for payload field '%s'...", field_name)
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def get_collection_info(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> Any:
        """Возвращает метаданные и статистику коллекции."""
        return self.client.get_collection(collection_name=collection_name)

    def count(self, collection_name: str = DEFAULT_COLLECTION_NAME, exact: bool = True) -> int:
        """Возвращает количество точек в коллекции."""
        res = self.client.count(collection_name=collection_name, exact=exact)
        return res.count

    def upsert_points(
        self,
        points: List[models.PointStruct],
        collection_name: str = DEFAULT_COLLECTION_NAME,
        wait: bool = False,
    ) -> Any:
        """Пакетная вставка / обновление точек в коллекции."""
        return self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
        )

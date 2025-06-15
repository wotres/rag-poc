import os
import logging
from typing import List, Optional

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

logger = logging.getLogger(__name__)

# Milvus 연결 정보
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# 임베딩 차원
VECTOR_DIM = 384

# 문서 청크 저장용 컬렉션 이름
DOCS_COLLECTION_NAME = "rag_docs_collection"


def init_milvus() -> Collection:
    """
    Milvus 서버에 연결하고, RAG 문서 청크 컬렉션을 생성하거나 로드합니다.
    반환값: docs_collection
    """
    # Milvus 서버 연결
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    # 컬렉션이 존재하면 로드, 없으면 생성
    if utility.has_collection(DOCS_COLLECTION_NAME):
        docs_col = Collection(DOCS_COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id",        dtype=DataType.INT64,       is_primary=True,  auto_id=True),
            FieldSchema(name="filename",  dtype=DataType.VARCHAR,     max_length=256,   is_primary=False),
            FieldSchema(name="chunk_id",  dtype=DataType.INT64,       is_primary=False),
            FieldSchema(name="chunk_text",dtype=DataType.VARCHAR,     max_length=2048,  is_primary=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,dim=VECTOR_DIM,   is_primary=False),
        ]
        schema = CollectionSchema(fields, description="RAG 문서 청크 저장용 컬렉션")
        docs_col = Collection(name=DOCS_COLLECTION_NAME, schema=schema)
        # 벡터 인덱스 생성
        docs_col.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        )
        docs_col.load()

    return docs_col


# 모듈 로드 시 초기화
_docs_collection: Collection = init_milvus()

def save_document_chunks_to_milvus(
    filename: str,
    chunks: List[str],
    embeddings: List[List[float]]
) -> None:
    """
    PDF 문서 청크 및 임베딩을 저장합니다.
    """
    chunk_ids = list(range(len(chunks)))
    filenames = [filename] * len(chunks)

    entities = [
        filenames,
        chunk_ids,
        chunks,
        embeddings,
    ]
    _docs_collection.insert(entities)
    _docs_collection.flush()


def search_top_k_doc_chunks(
    query_emb: List[float],
    top_k: int = 3,
    expr: Optional[str] = None
) -> List[dict]:
    """
    가장 유사한 상위 K개의 문서 청크를 검색합니다.
    반환값: [{"filename": ..., "chunk_id": ..., "chunk_text": ..., "distance": ...}, ...]
    """
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = _docs_collection.search(
        data=[query_emb],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["filename", "chunk_id", "chunk_text"]
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "filename": hit.entity.get("filename"),
            "chunk_id": hit.entity.get("chunk_id"),
            "chunk_text": hit.entity.get("chunk_text"),
            "distance": hit.distance
        })
    return hits


def delete_document_chunks_from_milvus(
    filename: str
) -> int:
    """
    지정된 파일명(filename)에 해당하는 모든 청크를 Milvus에서 삭제합니다.
    반환값: 삭제된 엔티티 수.
    """
    expr = f"filename == '{filename}'"
    res = _docs_collection.delete(expr)
    _docs_collection.flush()
    try:
        count = res.delete_count
    except AttributeError:
        count = len(res)
    logger.info(f"Milvus: '{{filename}}' 청크 {{count}}개 삭제됨.")
    return count

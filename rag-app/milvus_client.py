# milvus_client.py
import os

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# Milvus 연결 정보 (환경에 따라 수정)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# 1) 질문 저장용 컬렉션 이름 및 임베딩 차원
QUESTION_COLLECTION_NAME = "user_question_collection"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 임베딩 차원

# 2) RAG 문서 청크 저장용 컬렉션 이름
DOCS_COLLECTION_NAME = "rag_docs_collection"


def init_milvus():
    """
    Milvus 서버에 연결하고,
      1) 질문 저장용 컬렉션(QUESTION_COLLECTION_NAME) 생성 혹은 로드
      2) RAG 문서 청크 저장용 컬렉션(DOCS_COLLECTION_NAME) 생성 혹은 로드
    반환값: dict 형태로 { "questions": Collection, "docs": Collection }
    """
    # Milvus 서버 연결
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    # --- 1. 질문 컬렉션 초기화 ---
    if utility.has_collection(QUESTION_COLLECTION_NAME):
        question_collection = Collection(QUESTION_COLLECTION_NAME)
    else:
        q_fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=False,
                description="질문을 보낸 유저 아이디"
            ),
            FieldSchema(
                name="question",
                dtype=DataType.VARCHAR,
                max_length=1024,
                is_primary=False,
                description="질문 텍스트"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
                is_primary=False,
                description="질문 임베딩 벡터"
            ),
        ]
        q_schema = CollectionSchema(q_fields, description="사용자 질문 저장 컬렉션")
        question_collection = Collection(
            name=QUESTION_COLLECTION_NAME,
            schema=q_schema
        )
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        question_collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        question_collection.load()

    # --- 2. RAG 문서 청크 컬렉션 초기화 ---
    if utility.has_collection(DOCS_COLLECTION_NAME):
        docs_collection = Collection(DOCS_COLLECTION_NAME)
    else:
        d_fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="filename",
                dtype=DataType.VARCHAR,
                max_length=256,
                is_primary=False,
                description="원본 PDF 파일명"
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.INT64,
                is_primary=False,
                description="파일 내 청크 순번"
            ),
            FieldSchema(
                name="chunk_text",
                dtype=DataType.VARCHAR,
                max_length=2048,
                is_primary=False,
                description="청크 텍스트 (최대 2048자)"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
                is_primary=False,
                description="청크 임베딩 벡터"
            ),
        ]
        d_schema = CollectionSchema(d_fields, description="RAG 문서 청크 저장용 컬렉션")
        docs_collection = Collection(
            name=DOCS_COLLECTION_NAME,
            schema=d_schema
        )
        index_params_docs = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        docs_collection.create_index(
            field_name="embedding",
            index_params=index_params_docs
        )
        docs_collection.load()

    return {
        "questions": question_collection,
        "docs": docs_collection
    }


# 모듈 로딩 시, 미리 Milvus에 연결하여 두 개의 Collection 객체 획득
collections = init_milvus()
question_collection = collections["questions"]
docs_collection = collections["docs"]


def save_question_to_milvus(user_id: str, question: str, embedding: list[float]) -> None:
    """
    Milvus user_question_collection에 질문 레코드 삽입.
    """
    global question_collection
    entities = [
        [user_id],        # user_id 컬럼
        [question],       # question 컬럼
        [embedding],      # embedding 컬럼
    ]
    question_collection.insert(entities)
    question_collection.flush()


def save_document_chunks_to_milvus(
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]]
) -> None:
    """
    Milvus rag_docs_collection에 문서 청크 단위로 삽입.
    - filename: 업로드된 PDF 파일명 (예: "example.pdf")
    - chunks: ["텍스트 청크1", "텍스트 청크2", ...]
    - embeddings: [[emb1], [emb2], ...] (각 청크 당 VECTOR_DIM 길이 벡터)
    """
    global docs_collection
    # chunk_id, chunk_text, embedding 순서대로 entities 구성
    filenames = [filename] * len(chunks)
    chunk_ids = list(range(len(chunks)))
    chunk_texts = chunks
    embedding_vectors = embeddings

    entities = [
        filenames,         # filename 컬럼
        chunk_ids,         # chunk_id 컬럼
        chunk_texts,       # chunk_text 컬럼
        embedding_vectors  # embedding 컬럼
    ]
    docs_collection.insert(entities)
    docs_collection.flush()


def search_top_k_doc_chunks(
    query_emb: list[float],
    top_k: int = 3,
    expr: str = None
) -> list[dict]:
    """
    Milvus rag_docs_collection에서 query_emb와 가장 유사한 상위 K개의 청크를 찾아 반환.
    - query_emb: 리스트 형태의 임베딩 벡터
    - top_k: 반환할 청크 개수
    - expr: (선택) 필터 식. 예: "filename == 'example.pdf'"
    반환값: [
      {"filename": "...", "chunk_id": 0, "chunk_text": "...", "distance": 0.123},
      ...
    ]
    """
    global docs_collection
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = docs_collection.search(
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

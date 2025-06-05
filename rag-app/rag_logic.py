# rag_logic.py

import os
import pickle
import shutil
import fitz  # PyMuPDF
import numpy as np
import requests

# Milvus 저장/검색 함수 import
from milvus_client import (
    save_question_to_milvus,
    save_document_chunks_to_milvus,
    search_top_k_doc_chunks
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_PATH = "docs_data.pkl"

LLM_SERVICE_HOST = os.getenv("LLM_SERVICE_HOST", "0.0.0.0")
LLM_SERVICE_PORT = os.getenv("LLM_SERVICE_PORT", "8001")

# (선택) 로컬에 문서와 전체 임베딩을 백업하려면 아래 리스트 유지
docs: list[tuple[str, str]] = []
embeddings: list[np.ndarray] = []

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        saved = pickle.load(f)
        for path, content, emb in saved:
            if os.path.exists(path):
                docs.append((path, content))
                embeddings.append(np.array(emb))


def save_data():
    """
    로컬용 pickle로 docs와 embeddings 백업.
    (Milvus에 이미 저장되지만, 로컬에도 유지하고 싶다면)
    """
    to_save = [(path, text, emb.tolist()) for (path, text), emb in zip(docs, embeddings)]
    with open(DATA_PATH, "wb") as f:
        pickle.dump(to_save, f)


def list_documents() -> str:
    return "\n".join(sorted(f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")))


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def fetch_embeddings(texts: list[str]) -> np.ndarray:
    """
    LLM 서비스에 embedding 요청.
    - texts: 리스트 형태의 텍스트
    - 반환: numpy.ndarray(shape=(len(texts), VECTOR_DIM))
    """
    payload = {"model": "all-MiniLM-L6-v2", "input": texts}
    url = f"http://{LLM_SERVICE_HOST}:{LLM_SERVICE_PORT}/v1/embeddings"
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return np.array([item["embedding"] for item in data])


def save_and_index(file_path: str) -> str:
    """
    1) uploads 폴더로 복사
    2) PDF에서 텍스트 추출 → 텍스트가 없으면 에러 리턴
    3) 추출된 텍스트를 500자 단위로 쪼개서 fetch_embeddings(chunks) 호출
    4) Milvus rag_docs_collection에 chunk별로 저장
    5) (선택) 로컬 docs, embeddings 리스트에도 저장하고 pickle 백업
    """
    filename = os.path.basename(file_path)
    if not filename.endswith(".pdf"):
        return "PDF 파일만 지원합니다."

    dest = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(dest):
        return f"'{filename}' 이미 업로드됨."

    # 1) uploads 디렉토리로 복사
    shutil.copy(file_path, dest)

    # 2) PDF → 전체 텍스트 추출
    content = extract_text_from_pdf(dest)
    if not content.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."

    # 3) 500자 단위로 텍스트를 쪼개기
    chunk_size = 500
    chunks = [
        content[i : i + chunk_size]
        for i in range(0, len(content), chunk_size)
        if content[i : i + chunk_size].strip()
    ]

    # 4) 각 청크별로 embedding 생성 (batch 호출)
    try:
        chunk_embeddings = fetch_embeddings(chunks)  # shape = (len(chunks), VECTOR_DIM)
    except Exception as e:
        return f"임베딩 생성 중 오류: {e}"

    # 5) Milvus에 청크별로 삽입
    try:
        save_document_chunks_to_milvus(filename, chunks, chunk_embeddings.tolist())
    except Exception as e:
        return f"Milvus 저장 중 오류: {e}"

    # 6) (선택) 로컬 메모리/피클에도 저장 (문서 대표 벡터: 청크 임베딩 평균)
    docs.append((dest, content))
    embeddings.append(np.mean(chunk_embeddings, axis=0))
    save_data()

    return f"'{filename}' 업로드 및 500자 단위 인덱싱 완료!"


def delete_document(filename: str) -> str:
    """
    로컬 uploads 폴더에서 PDF 삭제 및 로컬 인덱스 제거.
    (Milvus 상에는 그대로 남아 있음 — 필요 시 별도 삭제 로직 구현)
    """
    fp = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(fp):
        return f"'{filename}' 파일을 찾을 수 없습니다."
    for i, (path, _) in enumerate(docs):
        if os.path.basename(path) == filename:
            docs.pop(i)
            embeddings.pop(i)
            break
    else:
        return f"'{filename}' 인덱스에 없음."
    os.remove(fp)
    save_data()
    return f"'{filename}' 삭제 완료."


def query_rag(
    query: str,
    llm_fn,
    selected_filename: str = None,
    user_id: str = None
) -> str:
    """
    RAG 기반 질의 함수.
    - user_id: 질의를 보낸 유저 ID (Milvus 질문 저장용)
    """
    # 1) 질문 embedding + Milvus 질문 저장
    try:
        q_emb = fetch_embeddings([query])[0]
        if user_id:
            save_question_to_milvus(user_id, query, q_emb.tolist())
    except Exception as e:
        print(f"[Warning] Milvus 질문 저장 실패: {e}")

    # 2) Milvus에서 상위 K개 청크 검색 (필요 시 selected_filename 필터 적용)
    expr = None
    if selected_filename:
        # 예: filename 필드를 정확하게 매칭해서 필터
        expr = f"filename == '{selected_filename}'"
    try:
        top_chunks = search_top_k_doc_chunks(q_emb.tolist(), top_k=3, expr=expr)
    except Exception as e:
        print(f"[Warning] 문서청크 Milvus 검색 실패: {e}")
        top_chunks = []

    # 3) LLM에게 청크들을 DocumentContext로 주고, 질문에 답 요청
    if not top_chunks:
        return llm_fn(f"(문서 없음 혹은 Milvus 검색 오류)\n\n질문: {query}\n답변:")

    # 여러 개의 청크를 하나의 문자열로 합쳐서 prompt 생성
    context_text = "\n\n".join(
        [f"### Chunk {c['chunk_id']} ({c['filename']}):\n{c['chunk_text']}" for c in top_chunks]
    )
    prompt = f"# DocumentContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    return llm_fn(prompt)

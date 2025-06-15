import os
import shutil
import logging
from typing import List

import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from milvus_registry import (
    save_document_chunks_to_milvus,
    delete_document_chunks_from_milvus,
    search_top_k_doc_chunks
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 환경 설정 ---
USERS = {
    "manager1": {"password": "manager1", "role": "manager", "group": "A"},
    "user1":    {"password": "user1",    "role": "user",    "group": "A"},
    "manager2": {"password": "manager2", "role": "manager", "group": "B"},
    "user2":    {"password": "user2",    "role": "user",    "group": "B"},
}
UPLOAD_DIR = "uploads"
LLM_HOST = os.getenv("LLM_SERVICE_HOST", "localhost")
LLM_PORT = os.getenv("LLM_SERVICE_PORT", "8888")

# 업로드 디렉토리 초기화
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Pydantic 모델 ---
class SearchRequest(BaseModel):
    query: str
    filename: str

class DeleteRequest(BaseModel):
    username: str
    filename: str

class DeleteResponse(BaseModel):
    message: str
    docs: List[str]
    role: str

# --- FastAPI 앱 ---
app = FastAPI()

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/documents", response_model=List[str])
def list_docs(user: str = Form(...)) -> List[str]:
    """해당 그룹의 PDF 목록을 반환"""
    user_info = USERS.get(user)
    if not user_info:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "유효하지 않은 사용자입니다.")
    group_dir = os.path.join(UPLOAD_DIR, user_info["group"])
    os.makedirs(group_dir, exist_ok=True)
    return [f for f in os.listdir(group_dir) if f.lower().endswith(".pdf")]

@app.post("/documents/upload")
async def upload_document(
    username: str = Form(...),
    file: UploadFile = File(...)
) -> dict:
    """PDF 업로드 후 Milvus 인덱싱"""
    user_info = USERS.get(username)
    if not user_info:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "유효하지 않은 사용자입니다.")
    group_dir = os.path.join(UPLOAD_DIR, user_info["group"])
    os.makedirs(group_dir, exist_ok=True)

    temp_path = os.path.join(group_dir, file.filename)
    content_bytes = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content_bytes)

    # PDF에서 텍스트 추출
    doc = fitz.open(temp_path)
    content = "".join(page.get_text() for page in doc)
    if not content.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "PDF에서 텍스트를 추출할 수 없습니다.")

    # 청크 분할
    chunk_size = 500
    chunks = [
        content[i : i + chunk_size]
        for i in range(0, len(content), chunk_size)
        if content[i : i + chunk_size].strip()
    ]

    # 임베딩 생성
    try:
        emb_resp = requests.post(
            f"http://{LLM_HOST}:{LLM_PORT}/v1/embeddings",
            json={"model": "all-MiniLM-L6-v2", "input": chunks}
        )
        emb_resp.raise_for_status()
        data = emb_resp.json().get("data", [])
        chunk_embs = [item["embedding"] for item in data]
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"임베딩 생성 오류: {e}")

    # Milvus 저장
    try:
        save_document_chunks_to_milvus(file.filename, chunks, chunk_embs)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Milvus 저장 오류: {e}")

    return {"message": f"'{file.filename}' 업로드 및 인덱싱 완료."}

@app.post("/documents/delete", response_model=DeleteResponse)
def delete_document(request: DeleteRequest) -> DeleteResponse:
    """PDF 및 Milvus 인덱스 삭제"""
    user_info = USERS.get(request.username)
    if not user_info:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "유효하지 않은 사용자입니다.")

    group_dir = os.path.join(UPLOAD_DIR, user_info["group"])
    target = os.path.join(group_dir, request.filename)
    if not os.path.exists(target):
        docs_list = [f for f in os.listdir(group_dir) if f.lower().endswith(".pdf")]
        return DeleteResponse(
            message=f"'{request.filename}' 파일이 존재하지 않습니다.",
            docs=docs_list,
            role=user_info["role"]
        )

    # Milvus 및 로컬 파일 삭제
    try:
        delete_document_chunks_from_milvus(request.filename)
    except Exception as e:
        logger.warning(f"[Milvus 삭제 실패] {e}")
    os.remove(target)

    docs_list = [f for f in os.listdir(group_dir) if f.lower().endswith(".pdf")]
    return DeleteResponse(
        message=f"'{request.filename}' 삭제 완료.",
        docs=docs_list,
        role=user_info["role"]
    )

@app.post("/embeddings/search")
def search_embeddings(request: SearchRequest) -> dict:
    """RAG: 유사한 문서 청크 검색"""
    try:
        resp = requests.post(
            f"http://{LLM_HOST}:{LLM_PORT}/v1/embeddings",
            json={"model": "all-MiniLM-L6-v2", "input": [request.query]}
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        query_emb = data[0]["embedding"]
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"임베딩 생성 오류: {e}")

    try:
        hits = search_top_k_doc_chunks(query_emb, top_k=3, expr=f"filename == '{request.filename}'")
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Milvus 검색 오류: {e}")

    return {"chunks": hits}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)

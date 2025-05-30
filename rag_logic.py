import os
import pickle
import shutil
import fitz  # PyMuPDF
import numpy as np
import requests  # 외부 mock 서버 호출용

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_PATH = "docs_data.pkl"  # docs + embeddings 같이 저장

# mock LLM/임베딩 서버 주소
MOCK_BASE = os.getenv("MOCK_SERVER", "http://localhost:8001")

# docs: List[(path, content)]
# embeddings: List[np.ndarray]
docs: list[tuple[str, str]] = []
embeddings: list[np.ndarray] = []

# 기존 데이터 복원
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        saved = pickle.load(f)
        for path, content, emb in saved:
            if os.path.exists(path):
                docs.append((path, content))
                embeddings.append(np.array(emb))


def save_data():
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
    mock 서버의 /v1/embeddings 엔드포인트를 호출하여 임베딩을 가져옴
    """
    payload = {"model": "all-MiniLM-L6-v2", "input": texts}
    resp = requests.post(f"{MOCK_BASE}/v1/embeddings", json=payload)
    resp.raise_for_status()
    data = resp.json()["data"]
    # 리스트 순서대로 임베딩 추출
    return np.array([item["embedding"] for item in data])


def save_and_index(file_path: str) -> str:
    filename = os.path.basename(file_path)
    if not filename.endswith(".pdf"):
        return "PDF 파일만 지원합니다."
    dest = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(dest):
        return f"'{filename}' 이미 업로드됨."
    shutil.copy(file_path, dest)
    content = extract_text_from_pdf(dest)
    if not content.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."
    # mock 서버로 임베딩 요청
    emb = fetch_embeddings([content])[0]
    docs.append((dest, content))
    embeddings.append(emb)
    save_data()
    return filename


def delete_document(filename: str) -> str:
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


def query_rag(query: str, llm_fn, selected_filename: str = None):
    if not docs:
        return llm_fn(f"(문서 없음)\n\n질문: {query}\n답변:")
    if selected_filename:
        for (path, content), emb in zip(docs, embeddings):
            if os.path.basename(path) == selected_filename:
                prompt = f"# DocumentContext:\n{content}\n\nQuestion: {query}\nAnswer:"
                return llm_fn(prompt)
        return f"'{selected_filename}' 문서를 찾을 수 없습니다."
    # 전체에서 최근접 문서 1개 찾아서 RAG
    # 쿼리 임베딩 fetch
    q_emb = fetch_embeddings([query])[0]
    dists = np.linalg.norm(np.stack(embeddings) - q_emb, axis=1)
    idx = int(np.argmin(dists))
    content = docs[idx][1]
    prompt = f"# DocumentContext:\n{content}\n\nQuestion: {query}\nAnswer:"
    return llm_fn(prompt)

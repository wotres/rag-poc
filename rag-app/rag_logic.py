import os
import pickle
import shutil
import fitz  # PyMuPDF
import numpy as np
import requests

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_PATH = "docs_data.pkl"

LLM_SERVICE_HOST = os.getenv("LLM_SERVICE_HOST", "http://0.0.0.0")
LLM_SERVICE_PORT = os.getenv("LLM_SERVICE_PORT", "8001")

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
    to_save = [(path, text, emb.tolist()) for (path, text), emb in zip(docs, embeddings)]
    with open(DATA_PATH, "wb") as f:
        pickle.dump(to_save, f)


def list_documents() -> str:
    return "\n".join(sorted(f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")))


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def fetch_embeddings(texts: list[str]) -> np.ndarray:
    payload = {"model": "all-MiniLM-L6-v2", "input": texts}
    url = f"{LLM_SERVICE_HOST}:{LLM_SERVICE_PORT}/v1/embeddings"
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()["data"]
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
    q_emb = fetch_embeddings([query])[0]
    dists = np.linalg.norm(np.stack(embeddings) - q_emb, axis=1)
    idx = int(np.argmin(dists))
    content = docs[idx][1]
    prompt = f"# DocumentContext:\n{content}\n\nQuestion: {query}\nAnswer:"
    return llm_fn(prompt)

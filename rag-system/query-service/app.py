import os
import requests                                      # ← 추가
import numpy as np                                   # ← 추가
import httpx
from fastapi import FastAPI, HTTPException, status   # ← HTTPException, status 추가
from pydantic import BaseModel

# 환경변수
USERS = {
    "manager1": {"password": "manager1",  "role": "manager", "group": "A"},
    "user1":    {"password": "user1",     "role": "user",    "group": "A"},
    "manager2": {"password": "manager2",  "role": "manager", "group": "B"},
    "user2":    {"password": "user2",     "role": "user",    "group": "B"}
}

DOCUMENT_SERVICE_HOST = os.getenv("DOCUMENT_SERVICE_HOST", "localhost")
DOCUMENT_SERVICE_PORT = os.getenv("DOCUMENT_SERVICE_PORT", "8002")
LLM_HOST = os.getenv("LLM_SERVICE_HOST", "localhost")  # ← 변수명 통일
LLM_PORT = os.getenv("LLM_SERVICE_PORT", "8888")       # ← 변수명 통일

UPLOAD_DIR = "uploads"
NO_SELECTION_LABEL = "선택하지 않음"
USE_MOCK = True

app = FastAPI()
client = httpx.AsyncClient()

class LoginRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    username: str
    query: str
    selected_doc: str = None

def get_group_dir(group: str) -> str:
    dir_path = os.path.join(UPLOAD_DIR, group)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

@app.post("/login")
def login(data: LoginRequest):
    username = data.username
    password = data.password
    user = USERS.get(username)
    if not user or user["password"] != password:
        # HTTPException과 status를 import 했으니 이제 사용 가능
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 잘못되었습니다."
        )

    role = user["role"]
    dir_path = get_group_dir(user["group"])
    docs = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    return {"role": role, "docs": docs}


def call_llm(query: str) -> str:
    if USE_MOCK:
        response = requests.post(
            f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions",  # ← 호스트/포트 변수 수정
            json={"model": "mock-model", "messages": [{"role": "user", "content": query}]}
        )
        return response.json()["choices"][0]["message"]["content"]
    else:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]    # ← prompt → query
        )
        return completion.choices[0].message.content


@app.post("/query")
async def run_query(data: QueryRequest):
    query = data.query.strip()
    if not query:
        return "질의가 비어있습니다."

    selected_doc = data.selected_doc
    is_rag = selected_doc and selected_doc != NO_SELECTION_LABEL

    if not is_rag:
        return call_llm(query)

    # RAG flow
    # 1) embedding 생성
    payload = {"model": "all-MiniLM-L6-v2", "input": [query]}
    emb_url = f"http://{LLM_HOST}:{LLM_PORT}/v1/embeddings"
    emb_resp = requests.post(emb_url, json=payload).json()
    # 만약 OpenAI 호환 API 라면:
    # query_embedding = np.array(emb_resp["data"][0]["embedding"])
    # 아니면 직접 리스트 반환한다고 가정:
    query_embedding = np.array(emb_resp[0]["embedding"])

    # 2) Milvus 에서 top-k 검색 (함수 import/정의 필요)
    # from your_milvus_module import search_top_k_doc_chunks
    try:
        top_chunks = search_top_k_doc_chunks(
            query_emb=query_embedding.tolist(),
            top_k=3,
            expr=f"filename == '{selected_doc}'"
        )
    except Exception as e:
        print(f"[Warning] Milvus 검색 실패: {e}")
        top_chunks = []

    if not top_chunks:
        return "유사한 문서가 없습니다. LLM 응답:\n" + call_llm(query)

    # 3) context 합치고
    context = "\n\n---\n\n".join(top_chunks)
    prompt = (
        "아래는 관련 문서의 일부입니다. 문맥을 참고하여 질문에 답해주세요.\n\n"
        f"== 문서 컨텍스트 ==\n{context}\n\n"
        f"== 질문 ==\n{query}\n\n"
        "== 답변 ==\n"
    )
    # 4) 최종 LLM 호출
    return call_llm(prompt)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)

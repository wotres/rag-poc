from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn

# 실제 임베딩을 위해 sentence-transformers 사용
from sentence_transformers import SentenceTransformer

app = FastAPI()

# --- Chat Completions 모델용 ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    last_user_msg = [m.content for m in req.messages if m.role == "user"][-1]
    dummy_response = (
        "(Mock 응답) 참고한 RAG 파일 내용은 아래와 같습니다\n"
        f"{last_user_msg}\n안녕하세요?\n"
    )
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": dummy_response
            }
        }]
    }

# --- Embeddings 모델용 ---
class EmbeddingRequest(BaseModel):
    model: str
    # OpenAI API와 호환되도록 input 은 str 또는 List[str]
    input: Union[str, List[str]]

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: dict

# SentenceTransformer 모델 로드 (서버 시작 시 한 번만)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(req: EmbeddingRequest):
    # input 을 리스트로 통일
    texts = [req.input] if isinstance(req.input, str) else req.input
    # 실제 임베딩 계산 (numpy array → list)
    embeddings = embedder.encode(texts).tolist()

    # 응답 포맷 구성
    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]
    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage={"prompt_tokens": len(texts), "total_tokens": len(texts)}
    )

if __name__ == "__main__":
    uvicorn.run("mock_llm_server:app", host="0.0.0.0", port=8001, reload=True)

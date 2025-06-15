from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn

from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

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
        "(Mock 응답)\n"
        f"LLM 응답 메시지: 아래와 같은 질문을 주셨습니다.\n"
        f"{last_user_msg}"
    )
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": dummy_response
            }
        }]
    }

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: dict

embedder = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    embeddings = embedder.encode(texts).tolist()
    
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
    import uvicorn
    uvicorn.run("mock_llm_server:app", host="0.0.0.0", port=8888, reload=True)

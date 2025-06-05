import os
import time
import shutil
import requests

import gradio as gr
from fastapi import FastAPI

# RAG 로직에서 쓰던 함수들
from rag_logic import save_and_index, delete_document, fetch_embeddings

# Milvus에 질문 저장 및 검색 함수
from milvus_client import save_question_to_milvus, search_top_k_doc_chunks

USERS = {
    "manager1": {"password": "manager1",  "role": "manager", "group": "A"},
    "user1":    {"password": "user1",     "role": "user",    "group": "A"},
    "manager2": {"password": "manager2","role": "manager", "group": "B"},
    "user2":    {"password": "user2",   "role": "user",    "group": "B"}
}

USE_MOCK = True
UPLOAD_DIR = "uploads"
NO_SELECTION_LABEL = "선택하지 않음"
LLM_SERVICE_HOST = os.getenv("LLM_SERVICE_HOST", "0.0.0.0")
LLM_SERVICE_PORT = os.getenv("LLM_SERVICE_PORT", "8001")


def call_llm(prompt: str) -> str:
    """
    내부 모의(Mock) LLM 또는 실제 OpenAI API를 호출하여 응답을 받습니다.
    """
    if USE_MOCK:
        response = requests.post(
            f"http://{LLM_SERVICE_HOST}:{LLM_SERVICE_PORT}/v1/chat/completions",
            json={"model": "mock-model", "messages": [{"role": "user", "content": prompt}]}
        )
        return response.json()["choices"][0]["message"]["content"]
    else:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content


def get_group_dir(group: str) -> str:
    """
    사용자의 그룹(A/B 등)에 따라 업로드 폴더를 분리하여 관리합니다.
    """
    dir_path = os.path.join(UPLOAD_DIR, group)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_current_file_list(current_user):
    """
    현재 로그인된 사용자의 그룹 디렉토리 내 PDF 파일 목록을 반환합니다.

    반환값:
        - 파일 목록을 줄바꿈(\n)으로 이어붙인 문자열
        - Query 탭의 doc_selector 업데이트용 gr.update(...)
        - RAG 등록 탭의 rag_file_selector 업데이트용 gr.update(...)
    """
    if not current_user:
        # 로그인 안 된 상태라면 각 Dropdown에 “선택하지 않음”만 표시
        empty_update = gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
        return "", empty_update, empty_update

    user_info = USERS[current_user]
    dir_path = get_group_dir(user_info["group"])
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    # Dropdown choices 목록: 맨 앞에 “선택하지 않음” 추가
    choices = [NO_SELECTION_LABEL] + files

    # 파일 목록을 텍스트박스에 표시할 때에는 줄바꿈 문자열
    files_str = "\n".join(files) if files else ""
    # 두 개의 Dropdown을 동시에 업데이트
    update_doc_selector = gr.update(choices=choices, value=NO_SELECTION_LABEL)
    update_rag_selector = gr.update(choices=choices, value=NO_SELECTION_LABEL)
    return files_str, update_doc_selector, update_rag_selector


def upload_file(file_path, custom_name, current_user):
    """
    Gradio 파일 업로드 버튼과 연결됩니다.
    업로드된 PDF를 그룹 디렉토리에 복사한 뒤, save_and_index()로 500자 단위 인덱싱 + Milvus 저장을 수행합니다.
    """
    if not file_path:
        return "파일이 선택되지 않았습니다.", *get_current_file_list(current_user)

    filename = (custom_name.strip() if custom_name else os.path.basename(file_path))
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    user_info = USERS[current_user]
    group_dir = get_group_dir(user_info["group"])
    dest_path = os.path.join(group_dir, filename)
    shutil.copy(file_path, dest_path)

    # save_and_index(): 내부에서 PDF → 500자 단위 청크 → Milvus 저장 처리
    result = save_and_index(dest_path)
    time.sleep(0.2)

    if any(kw in result for kw in ["지원", "추출", "이미 업로드", "선택되지"]):
        return f"{result}", *get_current_file_list(current_user)

    return f"'{filename}' 업로드 및 인덱싱 완료!", *get_current_file_list(current_user)


def delete_file(filename, current_user):
    """
    Gradio 삭제 버튼과 연결됩니다.
    """
    if not filename or filename == NO_SELECTION_LABEL:
        return "삭제할 파일을 선택하세요.", *get_current_file_list(current_user)

    user_info = USERS[current_user]
    group_dir = get_group_dir(user_info["group"])
    file_path = os.path.join(group_dir, filename)

    # 로컬 파일 삭제 및 로컬 인덱스에서 제거
    result = delete_document(filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return result, *get_current_file_list(current_user)


def download_file(filename, current_user):
    """
    Gradio 다운로드 버튼과 연결됩니다.
    선택된 filename이 존재하면 해당 그룹 디렉토리의 절대 경로를 반환하여
    DownloadButton이 브라우저로 보내줄 수 있게 합니다.
    """
    if not filename or filename == NO_SELECTION_LABEL:
        return None

    user_info = USERS[current_user]
    group_dir = get_group_dir(user_info["group"])
    file_path = os.path.join(group_dir, filename)
    return file_path if os.path.exists(file_path) else None


def run_query(query, selected_doc, current_user):
    """
    Gradio 질문하기 버튼과 연결됩니다.
    """
    if not query:
        return "질문을 입력해주세요."

    # 1) query 임베딩 생성 및 Milvus user_question_collection에 저장
    try:
        q_emb = fetch_embeddings([query])[0]  # numpy array (VECTOR_DIM,)
        save_question_to_milvus(user_id=current_user, question=query, embedding=q_emb.tolist())
    except Exception as e:
        print(f"[Warning] Milvus 질문 저장 실패: {e}")
        # 임베딩 생성 실패 시 q_emb가 없으면 RAG 기능 사용 불가 → 그냥 LLM 호출만 수행
        q_emb = None

    # 2) selected_doc이 없으면 → 단순 LLM 호출
    if not selected_doc or selected_doc == NO_SELECTION_LABEL:
        answer = call_llm(query)
        emb_str = (
            f"쿼리 임베딩 (차원={len(q_emb)}):\n{q_emb.tolist()}\n\n"
            if q_emb is not None else ""
        )
        return emb_str + "LLM 응답:\n" + answer

    # 3) RAG 모드: Milvus에서 selected_doc 필터로 상위 3개 청크 검색
    top_chunks = []
    if q_emb is not None:
        expr = f"filename == '{selected_doc}'"
        try:
            results = search_top_k_doc_chunks(query_emb=q_emb.tolist(), top_k=3, expr=expr)
            top_chunks = results  # [{filename, chunk_id, chunk_text, distance}, ...]
        except Exception as e:
            print(f"[Warning] Milvus 문서청크 검색 실패: {e}")

    # 4) LLM prompt 생성: 상위 청크들을 문맥으로 제공
    if not top_chunks:
        answer = call_llm(query)
        emb_str = (
            f"쿼리 임베딩 (차원={len(q_emb)}):\n{q_emb.tolist()}\n\n"
            if q_emb is not None else ""
        )
        return emb_str + "RAG 검색 결과가 없어서, 일반 LLM 응답:\n" + answer

    # 5) 청크 3개를 묶어 하나의 텍스트 블록으로 정리
    context_parts = []
    for idx, chunk in enumerate(top_chunks, start=1):
        part = (
            f"--- 청크 {idx} ---\n"
            f"파일명: {chunk['filename']}\n"
            f"청크 ID: {chunk['chunk_id']}\n"
            f"유사도(거리): {chunk['distance']:.6f}\n"
            f"청크 텍스트:\n{chunk['chunk_text']}\n"
        )
        context_parts.append(part)
    context_text = "\n".join(context_parts)

    # 6) 최종 LLM prompt: “DocumentContext” 아래에 모든 청크를 붙이고 질문을 덧붙임
    prompt = f"# DocumentContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    answer = call_llm(prompt)

    # 7) 최종 출력 문자열 구성
    emb_str = f"쿼리 임베딩 (차원={len(q_emb)}):\n{q_emb.tolist()}\n\n"
    retrieved_str = (
        "=== Milvus에서 검색된 상위 3개 청크 ===\n"
        f"{context_text}\n\n"
    )
    answer_str = "=== LLM 최종 응답 ===\n" + answer

    return emb_str + retrieved_str + answer_str


def login_fn(username, password, state_user):
    """
    Gradio 로그인 로직
    """
    user = USERS.get(username)
    if user and user["password"] == password:
        role = user["role"]
        files_str, update_query, update_rag = get_current_file_list(username)
        return (
            f"로그인 성공: {role} 권한",
            gr.update(visible=False),  # 로그인 칼럼 숨기기
            gr.update(visible=True),   # 메인 칼럼 보여주기
            gr.update(visible=(role == "manager")),  # 관리자만 RAG 등록 탭 보이기
            username,
            files_str,
            update_query,
            update_rag
        )

    return (
        "아이디 또는 비밀번호가 잘못되었습니다.",
        None,
        None,
        None,
        state_user,
        "",
        gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL),
        gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
    )


with gr.Blocks() as demo:
    state_user = gr.State("")

    # ─── 로그인 화면 ───
    with gr.Column(visible=True) as login_col:
        gr.Markdown("# 🔐 로그인 필요")
        username_input = gr.Textbox(label="아이디")
        password_input = gr.Textbox(label="비밀번호", type="password")
        login_btn = gr.Button("로그인")
        login_status = gr.Textbox(label="로그인 상태", interactive=False)

    # ─── 로그인 후 메인 화면 ───
    with gr.Column(visible=False) as main_col:
        gr.Markdown("# 📄 RAG PDF 문서 질의 시스템")
        with gr.Tabs() as tabs:
            # ─ 질의 탭 ─
            with gr.TabItem("질의"):
                query_input = gr.Textbox(label="질문 입력", lines=2)
                doc_selector = gr.Dropdown(label="문서 선택", choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
                # download_btn = gr.DownloadButton(label="PDF 다운로드")
                query_btn = gr.Button("질문하기")
                query_output = gr.Textbox(label="답변 및 정보", lines=20)

            # ─ RAG 등록(업로드/삭제/다운로드) 탭 ─ (관리자만 보임)
            with gr.TabItem("RAG 등록", visible=False) as rag_tab:
                file_input = gr.File(label="PDF 문서 업로드", file_types=[".pdf"], file_count="single")
                name_input = gr.Textbox(label="저장할 파일명 (확장자 제외)")
                upload_btn = gr.Button("업로드 및 인덱싱")
                upload_output = gr.Textbox(label="업로드 결과", lines=2)

                # 업로드된 파일 목록(읽기 전용 Textbox)
                file_list = gr.Textbox(label="업로드된 문서 목록", lines=6, interactive=False)

                # RAG 등록 탭 전용: 삭제할 파일명 입력
                delete_input = gr.Textbox(label="삭제할 파일명 입력 (예: sample.pdf)")
                delete_btn = gr.Button("문서 삭제")
                delete_output = gr.Textbox(label="삭제 결과", lines=2)

                # ↓ 새로 추가된 부분: RAG 등록 탭에서 “문서 선택 후 다운로드” ↓
                rag_file_selector = gr.Dropdown(label="다운로드할 문서 선택", choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
                rag_download_btn = gr.DownloadButton(label="문서 다운로드")
                # ↑ 여기까지 추가된 부분 ↑

    # ─── 버튼 바인딩 ───
    # 로그인 버튼 동작 바인딩 (파일 목록, Query 탭 Dropdown, RAG 탭 Dropdown 세 개를 업데이트)
    login_btn.click(
        fn=login_fn,
        inputs=[username_input, password_input, state_user],
        outputs=[login_status, login_col, main_col, rag_tab, state_user,
                 # 파일 목록(Textbox), Query 탭 Dropdown, RAG 탭 Dropdown
                 file_list, doc_selector, rag_file_selector]
    )

    # 업로드 버튼 동작 바인딩 (업로드 → 파일 목록 + 두 Dropdown 업데이트)
    upload_btn.click(
        fn=upload_file,
        inputs=[file_input, name_input, state_user],
        outputs=[upload_output, file_list, doc_selector, rag_file_selector]
    )

    # 삭제 버튼 동작 바인딩 (삭제 → 파일 목록 + 두 Dropdown 업데이트)
    delete_btn.click(
        fn=delete_file,
        inputs=[delete_input, state_user],
        outputs=[delete_output, file_list, doc_selector, rag_file_selector]
    )

    # # 질의 탭 다운로드 버튼 바인딩 (Query 탭 전용)
    # download_btn.click(
    #     fn=download_file,
    #     inputs=[doc_selector, state_user],
    #     outputs=[download_btn]
    # )

    # RAG 등록 탭 다운로드 버튼 바인딩 (새로 추가된 부분)
    rag_download_btn.click(
        fn=download_file,
        inputs=[rag_file_selector, state_user],
        outputs=[rag_download_btn]
    )

    # 질문하기 버튼 바인딩
    query_btn.click(
        fn=run_query,
        inputs=[query_input, doc_selector, state_user],
        outputs=query_output
    )

app = FastAPI()
gr.mount_gradio_app(app, demo, path="")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)


@app.get("/health")
def health():
    return {"status": "ok"}

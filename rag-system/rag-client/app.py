import os
import gradio as gr
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# --- 상수 정의 ---
NO_SELECTION_LABEL = "선택하지 않음"
QUERY_SERVICE_HOST = os.getenv("QUERY_SERVICE_HOST", "localhost")
QUERY_SERVICE_PORT = os.getenv("QUERY_SERVICE_PORT", "8001")
DOCUMENT_SERVICE_HOST = os.getenv("DOCUMENT_SERVICE_HOST", "localhost")
DOCUMENT_SERVICE_PORT = os.getenv("DOCUMENT_SERVICE_PORT", "8002")

app = FastAPI()
client = httpx.AsyncClient()

current_status = {
    "file_list": "",
    "choices": [NO_SELECTION_LABEL]
}

# --- 로그인 함수 ---
async def login(username, password):
    payload = {"username": username, "password": password}
    response = await client.post(
        f"http://{QUERY_SERVICE_HOST}:{QUERY_SERVICE_PORT}/login",
        json=payload
    )

    if response.status_code == 200:
        data = response.json()                                # JSON 파싱 추가
        role = data.get("role", "")
        docs = data.get("docs", []) or []
        choices = [NO_SELECTION_LABEL] + docs
        file_list = "\n".join(docs) if docs else "업로드된 문서가 없습니다."

        current_status["file_list"] = file_list
        current_status["choices"] = choices

        return (
            f"로그인 성공: {role} 권한",
            username,
            gr.update(visible=False),                         # 로그인 페이지 숨기기
            gr.update(visible=True),                          # 메인 페이지 보이기
            gr.update(visible=(role == "manager")),           # 매니저 전용 컬럼 보이기/숨기기
            file_list,
            gr.update(choices=choices, value=NO_SELECTION_LABEL),
            gr.update(choices=choices, value=NO_SELECTION_LABEL)
        )
    else:
        return (
            "아이디 또는 비밀번호가 잘못되었습니다.",
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),                         # 매니저 컬럼 항상 숨기기
            "업로드된 문서가 없습니다.",
            gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL),
            gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
        )

# --- 질의 함수 (변경 없음) ---
async def run_query(query, selected_doc, username):
    payload = {"query": query, "selected_doc": selected_doc, "username": username}
    response = await client.post(
        f"http://{QUERY_SERVICE_HOST}:{QUERY_SERVICE_PORT}/query",
        json=payload
    )
    return response.text.replace("\\n", "\n").replace("\"", "")

# --- 파일 업로드 함수 ---
# --- 파일 업로드 함수 ---
async def upload_file(file, custom_name, username):
    if not file:
        return (
            "파일이 선택되지 않았습니다.",
            current_status["file_list"],
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
        )

    # Gradio File 컴포넌트는 로컬 경로(str)를 반환합니다.
    file_path = file  # file is already the temp‐file path

    # 업로드할 실제 파일명 결정
    filename = custom_name.strip() if custom_name else os.path.basename(file_path)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    data = {"username": username}
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            response = await client.post(
                f"http://{DOCUMENT_SERVICE_HOST}:{DOCUMENT_SERVICE_PORT}/documents/upload",
                data=data,
                files=files,
            )
    except Exception as e:
        return (
            f"파일 열기 오류: {e}",
            current_status["file_list"],
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
        )

    if response.status_code != 200:
        return (
            "파일 업로드 중 오류가 발생했습니다.",
            current_status["file_list"],
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
        )

    # 성공 시 상태 갱신
    current_status["choices"].append(filename)
    current_status["file_list"] += "\n" + filename
    return (
        f"{filename} 파일이 업로드 및 인덱싱 완료되었습니다.",
        current_status["file_list"],
        gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
        gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
    )
    
# --- 파일 삭제 함수 ---
async def delete_file(filename, username):
    if not filename or filename == NO_SELECTION_LABEL:
        return (
            "삭제할 파일을 선택하세요.",
            current_status["file_list"],
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL)
        )

    payload = {"filename": filename, "username": username}
    response = await client.post(
        f"http://{DOCUMENT_SERVICE_HOST}:{DOCUMENT_SERVICE_PORT}/documents/delete",
        json=payload
    )
    if response.status_code != 200:
        return (
            "파일 삭제 중 오류가 발생했습니다.",
            current_status["file_list"],
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL),
            gr.update(choices=current_status["choices"], value=NO_SELECTION_LABEL)
        )

    data = response.json()                                  # JSON 파싱
    docs = data.get("docs", []) or []
    message = data.get("message")
    choices = [NO_SELECTION_LABEL] + docs
    file_list = "\n".join(docs) if docs else "업로드된 문서가 없습니다."

    current_status["choices"] = choices
    current_status["file_list"] = file_list

    return (
        message,
        file_list,
        gr.update(choices=choices, value=NO_SELECTION_LABEL),
        gr.update(choices=choices, value=NO_SELECTION_LABEL)
    )

# --- 파일 다운로드 함수 (Gradio DownloadButton 용) ---
async def download_file(filename, username):
    if not filename or filename == NO_SELECTION_LABEL:
        return None
    payload = {"filename": filename, "username": username}
    response = await client.post(
        f"http://{DOCUMENT_SERVICE_HOST}:{DOCUMENT_SERVICE_PORT}/documents/download",
        json=payload
    )
    if response.status_code != 200:
        return None
    # Gradio DownloadButton은 (bytes, filename, mime) 튜플을 반환해야 합니다.
    return (response.content, filename, "application/pdf")

@app.get("/health")
def health():
    return {"status": "ok"}

# --- UI 빌드 ---
def build_ui():
    with gr.Blocks() as demo:
        state_user = gr.State("")

        # 로그인 화면
        with gr.Column(visible=True) as login_page:
            gr.Markdown("로그인 화면")
            username_input = gr.Textbox(label="아이디")
            password_input = gr.Textbox(label="비밀번호", type="password")
            login_btn = gr.Button("로그인")
            login_status_message = gr.Textbox(label="로그인 상태", interactive=False)

        # 메인 쿼리 화면
        with gr.Column(visible=False) as main_page:
            gr.Markdown("RAG 시스템")
            with gr.Tabs() as tabs:
                # 질의 탭
                with gr.TabItem("질의"):
                    query_input = gr.Textbox(label="질문 입력", lines=2)
                    doc_selector = gr.Dropdown(
                        label="문서 선택",
                        choices=[NO_SELECTION_LABEL],
                        value=NO_SELECTION_LABEL
                    )
                    query_btn = gr.Button("질문하기")
                    query_output = gr.Textbox(label="답변 및 정보", lines=20)

                # RAG 탭 (항상 보여주되, 내용만 권한별로 제어)
                with gr.TabItem("RAG 관리"):
                    with gr.Column(visible=False) as rag_manager_col:
                        file_input = gr.File(
                            label="PDF 문서 업로드",
                            file_types=[".pdf"],
                            file_count="single"
                        )
                        name_input = gr.Textbox(label="저장할 파일명 (확장자 제외)")
                        upload_btn = gr.Button("업로드 및 인덱싱")
                        upload_output_message = gr.Textbox(label="업로드 결과", lines=2)

                        file_list = gr.Textbox(label="업로드된 문서 목록", lines=6, interactive=False)

                        delete_input = gr.Textbox(label="삭제할 파일명 입력 (예: sample.pdf)")
                        delete_btn = gr.Button("문서 삭제")
                        delete_output_message = gr.Textbox(label="삭제 결과", lines=2)

                        rag_file_selector = gr.Dropdown(
                            label="다운로드할 문서 선택",
                            choices=[NO_SELECTION_LABEL],
                            value=NO_SELECTION_LABEL
                        )
                        download_btn = gr.DownloadButton(label="문서 다운로드")

        # 이벤트 연결
        login_btn.click(
            fn=login,
            inputs=[username_input, password_input],
            outputs=[
                login_status_message, state_user,
                # 로그인/메인 UI 전환
                login_page, main_page,
                # 매니저 컬럼(show/hide)
                rag_manager_col,
                # 파일 리스트·선택지 초기화
                file_list, doc_selector, rag_file_selector
            ]
        )

        query_btn.click(
            fn=run_query,
            inputs=[query_input, doc_selector, state_user],
            outputs=query_output
        )

        upload_btn.click(
            fn=upload_file,
            inputs=[file_input, name_input, state_user],
            outputs=[upload_output_message, file_list, doc_selector, rag_file_selector]
        )

        delete_btn.click(
            fn=delete_file,
            inputs=[delete_input, state_user],
            outputs=[delete_output_message, file_list, doc_selector, rag_file_selector]
        )

        download_btn.click(
            fn=download_file,
            inputs=[rag_file_selector, state_user],
            outputs=download_btn
        )

    return demo

demo = build_ui()
gr.mount_gradio_app(app, demo, path="")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

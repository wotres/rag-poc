import gradio as gr
from rag_logic import save_and_index, query_rag, delete_document
import requests
import os
import time
import shutil

# --- 사용자 계정 및 역할 정의 ---
USERS = {
    "manager1": {"password": "manager_pw",  "role": "manager", "group": "A"},
    "user1":    {"password": "user_pw",     "role": "user",    "group": "A"},
    "manager2": {"password": "manager2_pw","role": "manager", "group": "B"},
    "user2":    {"password": "user2_pw",   "role": "user",    "group": "B"}
}

USE_MOCK = True
UPLOAD_DIR = "uploads"  # Base 업로드 디렉토리
NO_SELECTION_LABEL = "선택하지 않음"

# --- LLM 호출 함수 ---
def call_llm(prompt: str) -> str:
    if USE_MOCK:
        response = requests.post(
            "http://127.0.0.1:8001/v1/chat/completions",
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

# --- 그룹 디렉토리 경로 생성 ---
def get_group_dir(group: str) -> str:
    dir_path = os.path.join(UPLOAD_DIR, group)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# --- 현재 그룹의 PDF 목록 조회 ---
def get_current_file_list(current_user):
    if not current_user:
        return "", gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
    user_info = USERS[current_user]
    dir_path = get_group_dir(user_info["group"])
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    choices = [NO_SELECTION_LABEL] + files
    return "\n".join(files), gr.update(choices=choices, value=NO_SELECTION_LABEL)

# --- 파일 업로드 및 인덱싱 ---
def upload_file(file_path, custom_name, current_user):
    if not file_path:
        return "❗ 파일이 선택되지 않았습니다.", *get_current_file_list(current_user)
    filename = (custom_name.strip() if custom_name else os.path.basename(file_path))
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    user_info = USERS[current_user]
    group_dir = get_group_dir(user_info["group"])
    dest_path = os.path.join(group_dir, filename)
    shutil.copy(file_path, dest_path)
    result = save_and_index(dest_path)
    time.sleep(0.2)
    if any(kw in result for kw in ["지원", "추출", "이미 업로드", "선택되지"]):
        return f"❗ {result}", *get_current_file_list(current_user)
    return f"✅ '{filename}' 업로드 및 인덱싱 완료!", *get_current_file_list(current_user)

# --- 문서 삭제 ---
def delete_file(filename, current_user):
    if not filename or filename == NO_SELECTION_LABEL:
        return "❗ 삭제할 파일을 선택하세요.", *get_current_file_list(current_user)
    user_info = USERS[current_user]
    group_dir = get_group_dir(user_info["group"])
    file_path = os.path.join(group_dir, filename)
    result = delete_document(filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return result, *get_current_file_list(current_user)

# --- 파일 다운로드 ---
def download_file(filename, current_user):
    if not filename or filename == NO_SELECTION_LABEL:
        return None
    user_info = USERS[current_user]
    group_dir = get_group_dir(user_info["group"])
    file_path = os.path.join(group_dir, filename)
    return file_path if os.path.exists(file_path) else None

# --- 질의 처리 ---
def run_query(query, selected_doc):
    if not selected_doc or selected_doc == NO_SELECTION_LABEL:
        return call_llm(query)
    return query_rag(query, call_llm, selected_doc)

# --- 로그인 처리 ---
def login_fn(username, password, state_user):
    user = USERS.get(username)
    if user and user["password"] == password:
        role = user["role"]
        files_str, dropdown = get_current_file_list(username)
        return (
            f"✅ 로그인 성공: {role} 권한",
            gr.update(visible=False),  # 로그인 화면 숨기기
            gr.update(visible=True),   # 메인 화면 보이기
            gr.update(visible=(role == "manager")),  # RAG 등록 탭 보이기 여부
            username,
            files_str,
            dropdown
        )
    # 로그인 실패
    return (
        "❌ 아이디 또는 비밀번호가 잘못되었습니다.",
        None,
        None,
        None,
        state_user,
        "",
        gr.update(choices=[NO_SELECTION_LABEL], value=NO_SELECTION_LABEL)
    )

# --- Gradio UI 구성 ---
with gr.Blocks() as demo:
    state_user = gr.State("")

    # 로그인 화면
    with gr.Column(visible=True) as login_col:
        gr.Markdown("# 🔐 로그인 필요")
        username_input = gr.Textbox(label="아이디")
        password_input = gr.Textbox(label="비밀번호", type="password")
        login_btn = gr.Button("로그인")
        login_status = gr.Textbox(label="로그인 상태", interactive=False)

    # 메인 화면
    with gr.Column(visible=False) as main_col:
        gr.Markdown("# 📄 RAG PDF 문서 질의 시스템")
        with gr.Tabs() as tabs:
            # 질의 탭
            with gr.TabItem("질의"):
                query_input = gr.Textbox(label="질문 입력", lines=2)
                doc_selector = gr.Dropdown(label="문서 선택", choices=[], value=None)
                download_btn = gr.DownloadButton(label="PDF 다운로드")
                query_btn = gr.Button("질문하기")
                query_output = gr.Textbox(label="답변", lines=10)
            # RAG 등록 (Manager 전용)
            with gr.TabItem("RAG 등록", visible=False) as rag_tab:
                file_input = gr.File(label="PDF 문서 업로드", file_types=[".pdf"], file_count="single")
                name_input = gr.Textbox(label="저장할 파일명 (확장자 제외)")
                upload_btn = gr.Button("업로드 및 인덱싱")
                upload_output = gr.Textbox(label="업로드 결과", lines=2)
                file_list = gr.Textbox(label="업로드된 문서 목록", lines=6)
                delete_input = gr.Textbox(label="삭제할 파일명 입력 (예: sample.pdf)")
                delete_btn = gr.Button("문서 삭제")
                delete_output = gr.Textbox(label="삭제 결과", lines=2)

    # 이벤트 연결
    login_btn.click(
        fn=login_fn,
        inputs=[username_input, password_input, state_user],
        outputs=[login_status, login_col, main_col, rag_tab, state_user, file_list, doc_selector]
    )
    upload_btn.click(
        fn=upload_file,
        inputs=[file_input, name_input, state_user],
        outputs=[upload_output, file_list, doc_selector]
    )
    delete_btn.click(
        fn=delete_file,
        inputs=[delete_input, state_user],
        outputs=[delete_output, file_list, doc_selector]
    )
    download_btn.click(
        fn=download_file,
        inputs=[doc_selector, state_user],
        outputs=[download_btn]
    )
    query_btn.click(
        fn=run_query,
        inputs=[query_input, doc_selector],
        outputs=query_output
    )

if __name__ == "__main__":
    demo.launch()

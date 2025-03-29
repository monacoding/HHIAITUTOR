from flask import Flask, render_template, request, jsonify, session
import os
import fitz  # PyMuPDF
import requests
import json
import uuid
import shutil
from datetime import datetime
import tiktoken  # 토큰 수 계산용

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret")

app.config['SESSION_TYPE'] = 'memory'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

UPLOAD_FOLDER = os.path.join("static", "uploads")
CHAT_LOG_FOLDER = "chat_logs"
CONTENT_FOLDER = os.path.join("static", "content")
DATA_FOLDER = "data"
LEARNED_CONTENT_FILE = os.path.join(CONTENT_FOLDER, "learned_content.txt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHAT_LOG_FOLDER, exist_ok=True)
os.makedirs(CONTENT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# 서버 시작 시 기존 데이터 삭제 (DATA_FOLDER 제외)
for folder in [UPLOAD_FOLDER, CHAT_LOG_FOLDER, CONTENT_FOLDER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

learned_files = {}

def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

def load_learned_content():
    if os.path.exists(LEARNED_CONTENT_FILE):
        with open(LEARNED_CONTENT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def save_learned_content(content):
    with open(LEARNED_CONTENT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

def count_tokens(text):
    # tiktoken을 사용해 토큰 수 계산 (GPT-3 인코딩 기준으로 추정)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_max_tokens(model):
    # 모델별 최대 토큰 수 (Ollama 공식 문서 미제공, 일반적인 값 가정)
    max_tokens = {
        "mistral": 8192,
        "llama3": 4096,
        "default": 4096  # 알 수 없는 모델의 기본값
    }
    return max_tokens.get(model, max_tokens["default"])

def update_learned_content():
    global learned_files
    current_files = {}
    learned_content = ""

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_FOLDER, filename)
            mod_time = os.path.getmtime(filepath)
            current_files[filepath] = mod_time

    if current_files != learned_files:
        print("파일 변경 감지, 재학습 시작...")
        for filepath in current_files:
            if filepath not in learned_files or current_files[filepath] != learned_files.get(filepath):
                print(f"학습: {filepath}")
                learned_content += f"\n\n--- {os.path.basename(filepath)} ---\n{extract_text_from_pdf(filepath)}"
        save_learned_content(learned_content)
        learned_files = current_files
    else:
        print("변경된 파일 없음, 기존 학습 내용 사용.")
        learned_content = load_learned_content()
    
    return learned_content

learned_content = update_learned_content()

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def save_content(session_id, key, content):
    filepath = os.path.join(CONTENT_FOLDER, f"{session_id}_{key}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

def load_content(session_id, key):
    filepath = os.path.join(CONTENT_FOLDER, f"{session_id}_{key}.txt")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def check_ollama_status():
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_ollama_models():
    if not check_ollama_status():
        return ["Ollama server not running"]
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["No models installed"]
        else:
            return ["Ollama server error"]
    except requests.exceptions.RequestException as e:
        return [f"Ollama connection failed: {str(e)}"]

def call_local_llm(messages, model=None):
    if not check_ollama_status():
        return "⚠️ Ollama 서버가 실행 중이지 않습니다. 터미널에서 'ollama serve'를 실행하세요."
    url = "http://localhost:11434/api/chat"
    available_models = get_ollama_models()
    if not model or model not in available_models:
        if "No models" in available_models[0] or "Ollama" in available_models[0]:
            return f"⚠️ 사용 가능한 모델이 없습니다: {available_models[0]}. 모델을 설치하세요 (예: 'ollama pull mistral')."
        model = available_models[0]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise Exception(f"로컬 LLM 호출 실패: {response.text}")
    except requests.exceptions.RequestException as e:
        return f"⚠️ Ollama 연결 실패: {str(e)}. Ollama가 실행 중인지 확인하세요."

@app.route("/", methods=["GET"])
def index():
    session_id = get_session_id()
    models = get_ollama_models()
    if "selected_model" not in session or session["selected_model"] not in models:
        session["selected_model"] = models[0] if models else "Ollama server not running"
    
    learned_content = load_learned_content()
    used_tokens = count_tokens(learned_content)
    max_tokens = get_max_tokens(session.get("selected_model", "default"))
    
    return render_template(
        "index.html",
        summary=load_content(session_id, "summary"),
        chat_history=json.loads(load_content(session_id, "chat_history") or "[]"),
        models=models,
        selected_model=session.get("selected_model"),
        used_tokens=used_tokens,
        max_tokens=max_tokens
    )

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": get_ollama_models()})

@app.route("/select_model", methods=["POST"])
def select_model():
    model = request.form["model"]
    available_models = get_ollama_models()
    if model in available_models:
        session["selected_model"] = model
        return jsonify({"selected_model": model})
    return jsonify({"error": "선택한 모델이 설치되지 않았습니다."}), 400

@app.route("/upload", methods=["POST"])
def upload_pdf():
    session_id = get_session_id()
    file = request.files["pdf_file"]
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        extracted_text = extract_text_from_pdf(filepath)
        save_content(session_id, "extracted_text", extracted_text)
        summary = summarize_text(extracted_text)
        save_content(session_id, "summary", summary)
        save_content(session_id, "chat_history", json.dumps([]))
        return jsonify({"summary": summary, "chat_history": []})
    return jsonify({"error": "파일 업로드 실패"}), 400

@app.route("/reset", methods=["POST"])
def reset_chat():
    session_id = get_session_id()
    save_content(session_id, "chat_history", json.dumps([]))
    return jsonify({"chat_history": []})

@app.route("/save", methods=["POST"])
def save_chat():
    session_id = get_session_id()
    filepath = os.path.join(CHAT_LOG_FOLDER, f"{session_id}.json")
    data_to_save = {
        "chat_history": json.loads(load_content(session_id, "chat_history") or "[]"),
        "extracted_text": load_content(session_id, "extracted_text"),
        "summary": load_content(session_id, "summary"),
        "selected_model": session.get("selected_model", "No models installed")
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    return jsonify({"message": "대화와 문서 내용이 저장되었습니다."})

@app.route("/load", methods=["POST"])
def load_chat():
    session_id = get_session_id()
    filepath = os.path.join(CHAT_LOG_FOLDER, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            save_content(session_id, "chat_history", json.dumps(data.get("chat_history", [])))
            save_content(session_id, "extracted_text", data.get("extracted_text", ""))
            save_content(session_id, "summary", data.get("summary", ""))
            session["selected_model"] = data.get("selected_model", "No models installed")
        return jsonify({
            "chat_history": data.get("chat_history", []),
            "summary": data.get("summary", ""),
            "selected_model": session["selected_model"]
        })
    return jsonify({"error": "저장된 대화가 없습니다."}), 404

def summarize_text(text):
    if not text.strip():
        return "⚠️ 문서에서 텍스트를 추출하지 못했습니다."
    messages = [
        {"role": "system", "content": "넌 조선산업 관련 기술문서를 쉽게 설명하는 요약 전문가야. 수학 수식은 LaTeX 형식(예: $Q_{\\text{actual}}$)으로 작성하고, 감정을 담아 친절하게 설명해줘 😄"},
        {"role": "user", "content": f"다음 기술 문서를 신입사원이 이해할 수 있도록 요약해줘:\n{text[:3000]}"}
    ]
    return call_local_llm(messages)

@app.route("/chat", methods=["POST"])
def chat():
    session_id = get_session_id()
    user_question = request.form["question"]
    chat_history = json.loads(load_content(session_id, "chat_history") or "[]")
    chat_history.append(("user", user_question))

    learned_content = load_learned_content()
    extracted_text = load_content(session_id, "extracted_text")
    combined_context = learned_content + (f"\n\n--- 업로드된 문서 ---\n{extracted_text}" if extracted_text.strip() else "")

    if combined_context.strip():
        messages = [
            {"role": "system", "content": "넌 조선산업 기술문서를 잘 이해하고 설명하는 AI 튜터야. 아래 제공된 학습된 내용과 업로드된 문서 내용만을 기반으로 답변하며, 수학 수식은 LaTeX 형식(예: $Q_{\\text{actual}} = C_d \\cdot A \\cdot \\sqrt{\\frac{2 \\cdot \\Delta P}{\\rho}}$)으로 작성해줘. 이전 대화 맥락을 기억하고, 항상 따뜻하게, 이모지와 함께 대답해줘! 🛳️😊"},
            {"role": "system", "content": f"📄 학습된 내용과 문서:\n{combined_context}"}
        ]
    else:
        messages = [
            {"role": "system", "content": "넌 친절하고 다재다능한 AI 튜터야. 수학 수식은 LaTeX 형식(예: $Q_{\\text{actual}} = C_d \\cdot A \\cdot \\sqrt{\\frac{2 \\cdot \\Delta P}{\\rho}}$)으로 작성하고, 이전 대화 맥락을 기억하며, 항상 따뜻하게, 이모지와 함께 대답해줘! 😊"}
        ]

    for speaker, text in chat_history[:-1]:
        role = "user" if speaker == "user" else "assistant"
        messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_question})

    answer = call_local_llm(messages)
    chat_history.append(("assistant", answer))
    save_content(session_id, "chat_history", json.dumps(chat_history))
    return jsonify({"chat_history": chat_history})

if __name__ == "__main__":
    app.run(debug=True, port=5050)
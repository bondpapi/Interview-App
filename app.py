import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_local_storage import LocalStorage

# --- Setup (env + client) ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to your local .env or Streamlit Secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Interview Practice App", page_icon="💼", layout="centered")

# --- Optional diagnostics (safe) ---
with st.expander("Diagnostics (safe)", expanded=False):
    source = "ENV" if os.getenv("OPENAI_API_KEY") else ("SECRETS" if "OPENAI_API_KEY" in st.secrets else "NONE")
    st.write(f"Key source: **{source}**  |  Present: **{bool(api_key)}**")

# --- Local storage persistence ---
storage = LocalStorage()

if "messages" not in st.session_state:
    stored = storage.getItem("interview_messages")
    if stored:
        try:
            st.session_state.messages = json.loads(stored)
        except json.JSONDecodeError:
            st.session_state.messages = []
    else:
        st.session_state.messages = []

def save_messages():
    storage.setItem("interview_messages", json.dumps(st.session_state.messages))

# --- Helpers ---
def load_prompt(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful interview preparation assistant."

def build_transcript_text(system_prompt: str) -> str:
    lines = ["--- Interview Practice Transcript ---", f"[System]: {system_prompt}", ""]
    for m in st.session_state.messages:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"[{who}]: {m['content']}")
    return "\n".join(lines)

# --- UI ---
st.title("💼 Interview Practice App")
with st.expander("About this app", expanded=False):
    st.markdown(
        """
Use this app to **practice interviews** with AI.
- Choose an interview style (default, technical, behavioral)
- Optionally paste a job description to tailor the session
- Messages are **saved** in your browser (local storage)
- Download a **transcript** anytime
        """
    )

# Sidebar settings
st.sidebar.header("⚙️ OpenAI Settings")
# (Optional) let user pick a model that works with v1.x client
model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=1000, value=300, step=50)

# Prompt templates
prompt_options = {
    "Default Prompt": "prompts/base_prompt.txt",
    "Technical Interview": "prompts/technical_prompt.txt",
    "Behavioral Interview": "prompts/behavioral_prompt.txt",
}
selected_prompt = st.selectbox("Choose Interview Style", list(prompt_options.keys()))
system_prompt = load_prompt(prompt_options[selected_prompt])

# Inputs
job_description = st.text_area("Job Description (Optional)", height=150,
                               placeholder="Paste a JD to tailor questions/answers…")
user_input = st.text_input("Your message (ask a question or say 'ask me a question')")

col1, col2, col3 = st.columns([1, 1, 1])
send_clicked = col1.button("Get Interview Response")
new_session_clicked = col2.button("Start New Session")
download_clicked = col3.button("Download Transcript")

# New session
if new_session_clicked:
    st.session_state.messages.clear()
    storage.removeItem("interview_messages")
    st.success("Started a new session.")

# Download transcript
if download_clicked:
    transcript = build_transcript_text(system_prompt)
    st.download_button("Download Now", data=transcript,
                       file_name="interview_transcript.txt", mime="text/plain")

# Chat logic (v1.x API)
if send_clicked:
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_messages()

        # build message list
        api_messages = [{"role": "system", "content": system_prompt}]
        if job_description.strip():
            api_messages.append({
                "role": "assistant",
                "content": f"Use the following job description to tailor the interview: {job_description.strip()}"
            })
        api_messages.extend(st.session_state.messages)

        try:
            with st.spinner("Generating AI response..."):
                completion = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                reply = completion.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": reply})
                save_messages()
        except Exception as e:
            st.error(f"OpenAI error: {e}")

# Render history
if st.session_state.messages:
    st.markdown("### Conversation")
    for m in st.session_state.messages:
        st.markdown(f"**{'You' if m['role']=='user' else 'Assistant'}:** {m['content']}")

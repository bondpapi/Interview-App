import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --- Load local .env for dev ---
load_dotenv()

# --- Normalize key into ENV so the SDK reads it the "standard" way ---
key = os.getenv("OPENAI_API_KEY")
if not key and "OPENAI_API_KEY" in st.secrets:
    # Force to str, strip whitespace/newlines just in case
    key = str(st.secrets["OPENAI_API_KEY"]).strip()

if not key:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to your local .env or Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = key  # <- set env var explicitly
client = OpenAI()  # <- no arg; SDK will read from env

with st.expander("Diagnostics (safe)", expanded=False):
    src = "ENV" if os.getenv("OPENAI_API_KEY") else ("SECRETS" if "OPENAI_API_KEY" in st.secrets else "NONE")
    tail = os.environ["OPENAI_API_KEY"][-4:]
    st.write(f"Key source: **{src}** | Present: **True** | Tail: **...{tail}**")

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": "..."}

# --- Helpers ---
def load_prompt(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful interview preparation assistant."

def build_transcript_text(system_prompt: str) -> str:
    lines = ["--- Interview Practice Transcript ---", f"[System]: {system_prompt}", ""]
    for m in st.session_state.messages:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"[{who}]: {m['content']}")
    return "\n".join(lines)

# --- UI: Title + About ---
st.title(" Interview Practice App")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
This app helps you **practice interviews** using OpenAI.
- Choose an **interview style** (default, technical, behavioral)
- Paste a **job description** (optional)
- Ask a question or have the AI ask *you* questions
- Your **conversation history** is saved for the session and can be **downloaded**

**Tip:** Use the sidebar to tune model creativity (temperature) and response length (max tokens).
        """
    )

# --- Sidebar: OpenAI settings ---
st.sidebar.header("⚙️ OpenAI Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=1000, value=300, step=50)

# --- Prompt selection ---
prompt_options = {
    "Default Prompt": "prompts/base_prompt.txt",
    "Technical Interview": "prompts/technical_prompt.txt",
    "Behavioral Interview": "prompts/behavioral_prompt.txt",
}
selected_prompt_name = st.selectbox("Choose Interview Style", list(prompt_options.keys()))
system_prompt = load_prompt(prompt_options[selected_prompt_name])

# --- Inputs ---
job_description = st.text_area("Job Description (Optional)", height=150, placeholder="Paste a JD to tailor questions/answers…")
user_input = st.text_input("Your message (ask a question or say 'ask me a question')")

col1, col2, col3 = st.columns([1, 1, 1])
send_clicked = col1.button("Get Interview Response")
new_session_clicked = col2.button("Start New Session")
download_clicked = col3.button("Download Transcript")

# --- New session ---
if new_session_clicked:
    st.session_state.messages.clear()
    st.success("Started a new session.")

# --- Download transcript ---
if download_clicked:
    transcript = build_transcript_text(system_prompt)
    st.download_button(
        "Download Now",
        data=transcript,
        file_name="interview_transcript.txt",
        mime="text/plain"
    )

# --- Chat logic ---
if send_clicked:
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # Add the user's message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Build message list for the API: system + history
        api_messages = [{"role": "system", "content": system_prompt}]

        # Include job description context if provided (as an assistant note to steer behavior)
        if job_description.strip():
            api_messages.append({
                "role": "assistant",
                "content": f"Use the following job description to tailor the interview: {job_description.strip()}"
            })

        # Add prior exchanges
        api_messages.extend(st.session_state.messages)

        try:
            with st.spinner("Generating AI response..."):
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                reply = completion.choices[0].message.content
                # Save assistant reply to history
                st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"OpenAI error: {e}")

# --- Render chat history ---
if st.session_state.messages:
    st.markdown("### Conversation")
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Assistant:** {m['content']}")


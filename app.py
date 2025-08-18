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

def render_token_cost(usage, in_rate, out_rate):
    # usage contains prompt_tokens, completion_tokens, total_tokens
    if not usage:
        return
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", prompt_tokens + completion_tokens)
    cost = (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate

    st.info(
        f"**Token usage** — Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total}\n\n"
        f"**Estimated cost:** ${cost:.6f} "
        f"(rates: ${in_rate}/1k prompt, ${out_rate}/1k completion)"
    )

def json_safe_load(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)

# --- UI ---
st.title("💼 Interview Practice App")
with st.expander("About this app", expanded=False):
    st.markdown(
        """
Use this app to **practice interviews** with AI.
- Choose an interview style (default, technical, behavioral)
- Optionally paste a job description to tailor the session
- Choose **output format** (normal text or structured JSON)
- Messages are saved in your browser (local storage)
- Download a **transcript** anytime
        """
    )

# Sidebar settings
st.sidebar.header("⚙️ OpenAI Settings")
model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
presence_penalty = st.sidebar.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
frequency_penalty = st.sidebar.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=2000, value=300, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("💲 Cost Estimator (set your rates)")
in_rate = st.sidebar.number_input("$/1k prompt tokens", min_value=0.0, value=0.0015, step=0.0001, format="%.6f")
out_rate = st.sidebar.number_input("$/1k completion tokens", min_value=0.0, value=0.0020, step=0.0001, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Judge (Optional)")
use_judge = st.sidebar.checkbox("Run LLM-as-Judge critique on assistant reply", value=False)

# Prompt templates + difficulty
prompt_options = {
    "Default Prompt": "prompts/base_prompt.txt",
    "Technical Interview": "prompts/technical_prompt.txt",
    "Behavioral Interview": "prompts/behavioral_prompt.txt",
}
selected_prompt = st.selectbox("Choose Interview Style", list(prompt_options.keys()))
difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
base_system = load_prompt(prompt_options[selected_prompt])

# Output format
out_format = st.selectbox(
    "Output format",
    ["Normal text", "JSON: QnA", "JSON: Evaluation"],
    index=0
)

# Build final system prompt with difficulty + format guidance
format_instructions = ""
if out_format == "JSON: QnA":
    format_instructions = (
        "Return ONLY valid JSON with this exact shape:\n"
        '{ "question": "<string>", "answer": "<string>" }'
    )
elif out_format == "JSON: Evaluation":
    format_instructions = (
        "Return ONLY valid JSON with this exact shape:\n"
        '{ "score": <integer 0-10>, "strengths": ["<string>", ...], "areas_to_improve": ["<string>", ...] }'
    )

system_prompt = (
    f"{base_system}\n\n"
    f"Difficulty: {difficulty}.\n"
    f"{format_instructions}"
    if format_instructions else
    f"{base_system}\n\nDifficulty: {difficulty}."
)

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

# Chat logic
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
                kwargs = dict(
                    model=model,
                    messages=api_messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    max_tokens=max_tokens,
                )
                # If user picked JSON format, ask model to strictly return JSON text
                # (We guide via system instructions; we still parse defensively.)
                completion = client.chat.completions.create(**kwargs)
                reply = completion.choices[0].message.content
                usage = getattr(completion, "usage", None)

                # Try to parse JSON if a JSON format was selected
                parsed = None
                parse_error = None
                if out_format != "Normal text":
                    parsed, parse_error = json_safe_load(reply)

                # Record assistant reply (store raw text for history)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                save_messages()

                # Render reply
                st.markdown("### Assistant")
                if parsed:
                    st.json(parsed)
                else:
                    st.write(reply)
                    if parse_error and out_format != "Normal text":
                        st.warning("Expected JSON but got text. Showing raw output. (Parsing error hidden for cleanliness.)")

                # Token usage + cost
                render_token_cost(usage, in_rate, out_rate)

                # Optional Judge pass
                if use_judge:
                    judge_instructions = (
                        "You are a strict interview evaluator. Critique the assistant's last answer for correctness, "
                        "clarity, depth, and relevance to the user's message and job description if present. "
                        "Provide 3-5 bullet points and a 0-10 score."
                    )
                    judge_messages = [
                        {"role": "system", "content": judge_instructions},
                        {"role": "user", "content": f"User's message: {user_input}"},
                        {"role": "user", "content": f"Assistant's reply: {reply}"},
                    ]
                    if job_description.strip():
                        judge_messages.append({"role": "user", "content": f"Job description: {job_description.strip()}"})

                    judge = client.chat.completions.create(
                        model=model,
                        messages=judge_messages,
                        temperature=0.2,
                        max_tokens=400
                    )
                    st.markdown("### Judge Critique (LLM-as-Judge)")
                    st.write(judge.choices[0].message.content)

        except Exception as e:
            st.error(f"OpenAI error: {e}")

# Render history
if st.session_state.messages:
    st.markdown("### Conversation")
    for m in st.session_state.messages:
        st.markdown(f"**{'You' if m['role']=='user' else 'Assistant'}:** {m['content']}")


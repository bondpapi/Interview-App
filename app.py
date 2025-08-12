import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Interview Practice App", page_icon="💼")

st.title("Interview Practice App")
st.write("Prepare for your interviews using AI-powered questions and feedback.")

# Function to load a prompt file
def load_prompt(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return "You are a helpful interview preparation assistant."

# Prompt selection dropdown
prompt_options = {
    "Default Prompt": "prompts/base_prompt.txt",
    "Technical Interview": "prompts/technical_prompt.txt",
    "Behavioral Interview": "prompts/behavioral_prompt.txt",
}
selected_prompt_name = st.selectbox("Choose Interview Style", list(prompt_options.keys()))
system_prompt = load_prompt(prompt_options[selected_prompt_name])

# OpenAI settings
st.sidebar.header("⚙️ OpenAI Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=1000, value=300, step=50)

# User input
job_description = st.text_area("Enter Job Description (Optional)", height=150)
user_question = st.text_input("Ask the AI interviewer a question")

# Action button
if st.button("Get Interview Response"):
    if not user_question.strip():
        st.warning("Please enter a question for the AI interviewer.")
    else:
        with st.spinner("Generating AI response..."):
            try:
                prompt = f"Job Description: {job_description}\n\nQuestion: {user_question}"
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.markdown("**AI Response:**")
                st.write(response.choices[0].message["content"])
            except Exception as e:
                st.error(f"Error: {e}")

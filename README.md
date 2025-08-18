# Interview Practice App

An app to stimulate professional Job Interview and help with Interview prep

## Overview
The **Interview Practice App** is a web-based tool that uses **prompt engineering** and the **OpenAI API** to help users prepare for job interviews.  
It allows for interactive interview simulations, tailored preparation strategies, and customizable question sets — making it a valuable resource for both technical and behavioral interview practice.

Later, you can continue expanding this app for personal use, professional coaching, or as a portfolio showcase.

---

## ✨ Features

### Core Features
- Single-page web application built with **Python (Streamlit)** or **JavaScript (Next.js)**.
- Integration with the **OpenAI API** for intelligent, context-aware responses.
- Customizable **system** and **user prompts** for different interview scenarios.
- Ability to simulate various interview types:
  - Technical interviews
  - Behavioral interviews
  - Role-specific Q&A sessions
- Option to analyze a job description and generate a tailored preparation plan.
- Adjustable AI settings (e.g., temperature, model selection) for different output styles.
- Basic security measures such as input validation.

---

## Advanced Features

- **Interview Styles**  
  Choose between *Default*, *Technical*, or *Behavioral* interview modes.

- **Difficulty Levels**  
  Pick **Easy**, **Medium**, or **Hard**.  
  The interviewer adapts tone, depth of questions, and follow-ups.

- **Custom Prompts & Job Descriptions**  
  Tailor sessions with your own prompts or paste a job description to guide responses.

- **Output Formats**  
  - Plain text  
  - JSON (Q&A format)  
  - JSON (Evaluation with score, strengths, areas to improve)

- **Judge Mode**  
  Let the AI critique your answers as an evaluator.

- **Jailbreak Tester**  
  Self-audit suspicious prompts and detect potential **prompt injection** attempts.

- **Cost Estimator**   
  Estimate token usage and approximate cost per session.

- **Conversation Persistence**  
  Sessions are stored in memory and can be downloaded as a transcript.

- **Image Generator** *(NEW!)*  
  - Generate **visual interview posters** from feedback.  
  - Create **custom diagrams** (e.g., system design sketches).  
  - Download generated images as PNG.  
  - Useful for making practice sessions **shareable and visual**.


---

## Tech Stack
- **Frontend:** Streamlit (Python) or Next.js (JavaScript)
- **Backend:** OpenAI API
- **Optional:** LangChain, Vector Databases, Image Generation APIs
- **Deployment:** Vercel, AWS, Azure, or similar

---

## Prerequisites
- Python or JavaScript knowledge
- Understanding of ChatGPT and OpenAI API
- Basic knowledge of front-end development
- An OpenAI API key

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/interview-practice-app.git
cd interview-practice-app
```
### 2. Install dependencies
For Python/Streamlit:
```bash
pip install -r requirements.txt
```
For JavaScript/Next.js:
```bash
npm install
```
### 3. Add your OpenAI API key
Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_api_key_here
```
### 4. Run the application
For Python/Streamlit:
```bash
streamlit run app.py
```
For JavaScript/Next.js:
```bash
npm run dev
```
### 5. Access the app
Open your browser and go to `http://localhost:8501` for Streamlit or `
http://localhost:3000` for Next.js.
---

## Author

**Michael Bond** — AI/ML Engineer | Data Scientist | Developer | Data Analyst

GitHub: https://github.com/bondpapi

LinkedIn: https://www.linkedin.com/in/bond-michael/
--- 
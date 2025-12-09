# 🎥 YouTube Summarizer with RAG Chatbot  
A powerful application that allows users to **summarize YouTube videos** and **chat with the actual content** using Retrieval-Augmented Generation (RAG).  
The system extracts a video's transcript, breaks it into searchable chunks, stores them in a vector database, and lets users ask questions directly from the content — ensuring accurate, context-aware responses.

---

## 🚀 Features

### ✅ Summarizer  
- Extracts full YouTube transcript  
- Generates **short**, **medium**, and **detailed** summaries  
- Handles long videos efficiently

### ✅ RAG Chatbot  
- Answers questions **based on real transcript content**  
- Uses embedding-based search for relevant chunks  
- Supports multi-turn conversation  
- Prevents hallucination by grounding all responses in transcript data

### ✅ Streamlit Web App  
- Clean and simple UI  
- Paste link → get summary → start chatting  
- Chat history panel  
- Error handling for invalid links or missing transcripts

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Backend  | Python |
| Web Framework | Streamlit |
| LLM | GPT / ChatGPT API / Llama (your choice) |
| RAG | FAISS / ChromaDB vector store |
| Embeddings | OpenAI / HuggingFace |
| Transcript Extraction | `youtube-transcript-api` |
| Environment Secrets | `.env` |

---

## 🏗️ System Architecture

1. User enters a YouTube video URL  
2. Transcript is fetched using the YouTube Transcript API  
3. Transcript is chunked using text splitter  
4. Embeddings are generated  
5. Vector database stores and indexes the chunks  
6. User asks a question  
7. Similar chunks are retrieved (RAG)  
8. LLM generates a grounded answer using retrieved context

---

## 📦 Installation

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/yt-summarizer-rag.git
cd yt-summarizer-rag

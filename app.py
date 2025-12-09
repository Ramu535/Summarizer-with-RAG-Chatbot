import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain import  PromptTemplate
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
             model_name="llama-3.1-8b-instant")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemma-3n-e2b-it")

# Session state initialization
if "final_text" not in st.session_state:
    st.session_state.final_text = ""

if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""

if "summary_content" not in st.session_state:
    st.session_state.summary_content = ""

if "vectors" not in st.session_state:
    st.session_state.vectors = None

st.set_page_config(page_title="AI Content Assistant", layout="wide", page_icon="🤖")

# Enhanced CSS for better appearance and distinct styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.8rem;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-response {
       
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #2196f3;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .summary-response {
    
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #4caf50;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        font-size: 1.2rem;
        line-height: 1.7;
        font-weight: 500;
    }
    
    .status-indicator {
        text-align: center;
        padding: 12px;
        border-radius: 25px;
        margin: 15px 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 2px solid #28a745;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.2);
    }
    
    .status-waiting {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        border: 2px solid #ffc107;
        box-shadow: 0 4px 8px rgba(255, 193, 7, 0.2);
    }
    
    .chatbot-enabled {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #17a2b8;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        color: #0c5460;
    }
    
    .summary-title {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 15px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .chat-title {
        color: #1565c0;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

os.environ['HUGGINGFACE_API_KEY']=os.getenv("HUGGINGFACE_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

summarizer_prompt = """
You are a professional summarizer.
Your task is to summarize the given text in a natural, readable way within 250 words. Use clear paragraphs for flow.
If the text covers multiple distinct topics or sections, use meaningful subheadings to separate them.
If appropriate, use bullet points for lists or key highlights, but only when it helps clarity. Do not add a main heading like 'Summary' — just start directly with the content.

Write the summary as if it's meant for someone who wants a quick but well-structured understanding of the original material.
"""

def extract_transcript_details(youtube_video_url):
    try:
        if "youtu.be" in youtube_video_url:
            video_id = youtube_video_url.split("/")[-1].split("?")[0]
        else:
            video_id = youtube_video_url.split("=")[1].split("&")[0]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript, video_id
    except Exception as e:
        raise e

def generate_gemini_content(text, prompt):
    response = model.generate_content([prompt, text])
    return response.text.strip()

def vector_embedding():
    try:
        # Clear previous vector store
        if "vector" in st.session_state:
            try:
                st.session_state.vector._store.close()
            except:
                pass
        
        # Create new embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.create_documents([st.session_state.final_text])
        st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state.vectors = True
        return True
    except Exception as e:
        st.error(f"Error creating vector embeddings: {str(e)}")
        return False

# Main title
st.markdown('<h1 class="main-title">🤖 AI Content Assistant</h1>', unsafe_allow_html=True)

# Status indicator
if st.session_state.final_text:
    st.markdown('<div class="status-indicator status-ready">✅ Content loaded - Chatbot is now ACTIVE!</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-indicator status-waiting">⏳ Load content from summarizer to activate chatbot</div>', unsafe_allow_html=True)

# Two-column layout with adjusted widths - Summarizer gets more space
 # Left for chatbot (1), right for summarizer (3)

# LEFT: CHATBOT PANEL
with st.sidebar:
    st.markdown('<div class="chat-title">💬 Chat with Content</div>', unsafe_allow_html=True)
    
    # Show content status
    if st.session_state.final_text:
        word_count = len(st.session_state.final_text.split())
        st.markdown(f'<div class="chatbot-enabled">📄Chatbot is Enabled!</div>', unsafe_allow_html=True)
        
        
    else:
        st.warning("⚠️ Load content first!")
    
    # Chat input - only show if content is loaded
    if st.session_state.final_text:
        input_prompt = st.text_input(
            "Ask a question:",
            placeholder="What would you like to know?",
            key="chat_input"
        )
        
        # Ask button
        col_ask, col_clear = st.columns([2, 1])
        with col_ask:
            if st.button("🚀 Ask", use_container_width=True):
                if input_prompt:
                    with st.spinner("🤔 AI is thinking..."):
                        try:
                            # Ensure vector embeddings exist
                            if not st.session_state.vectors:
                                vector_embedding()
                            st.write("Hi")
                            
                            document_chain = create_stuff_documents_chain(llm, prompt)
                            retriever = st.session_state.vector.as_retriever()
                            retrieval_chain = create_retrieval_chain(retriever, document_chain)
                            response = retrieval_chain.invoke({'input': input_prompt})
                            st.session_state.chat_response = response['answer']
                            st.write("Hello")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.info("Try enabling the chatbot first!")
                else:
                    st.warning("Please enter a question!")
        
        with col_clear:
            if st.button("🗑️", help="Clear", use_container_width=True):
                st.session_state.chat_response = ""
                st.rerun()
    
    # Display chat response
    if st.session_state.chat_response:
        st.markdown("#### 🤖 AI Response")
        st.markdown(f'<div class="chat-response">{st.session_state.chat_response}</div>', unsafe_allow_html=True)

# RIGHT: SUMMARIZER PANEL (Wider column)

st.markdown('<div class="summary-title">📄 Youtube/Website Summarizer</div>', unsafe_allow_html=True)
    
    # URL input
url = st.text_input(
        "Enter YouTube or Website URL:",
        placeholder="https://www.youtube.com/watch?v=... or https://example.com",
        key="url_input"
    )

if url:
        # Detect content type
    if "youtube.com" in url or "youtu.be" in url:
        st.markdown("### 📺 YouTube Video Detected")
        if st.button("📽️ Summarize YouTube Video", type="primary", use_container_width=True):
            with st.spinner("🎬 Processing video..."):
                try:
                    transcript_text, video_id = extract_transcript_details(url)
                        
                        # Show thumbnail
                    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")
                        
                        # Generate summary
                    summary = generate_gemini_content(transcript_text, summarizer_prompt)
                    st.session_state.final_text = transcript_text
                    st.session_state.summary_content = summary
                    st.session_state.vectors = None  # Reset vectors
                        
                    st.success(f"✅ Video processed! {len(transcript_text.split()):,} words extracted.")
                    st.balloons()  # Celebration effect
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
                    st.info("💡 Make sure the video has captions available")
        
    else:
        st.markdown("### 🌐 Website Detected")
        if st.button("🌐 Summarize Webpage", type="primary", use_container_width=True):
            with st.spinner("🌍 Processing webpage..."):
                try:
                    loader = WebBaseLoader(url)
                    text_documents = loader.load()
                    text = text_documents[0].page_content
                    st.session_state.final_text = text
                        
                        # Process in chunks
                    splitter = RecursiveCharacterTextSplitter(
                            chunk_size=2000,
                            chunk_overlap=100,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                    chunks = splitter.split_text(text)

                    summaries = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                        
                    for i, chunk in enumerate(chunks):
                        status_text.text(f"Processing chunk {i+1} of {len(chunks)}...")
                        try:
                            response = model.generate_content([summarizer_prompt, chunk])
                            summaries.append(response.text.strip())
                            progress_bar.progress((i + 1) / len(chunks))
                        except Exception as e:
                            st.warning(f"⚠️ Error in chunk {i+1}: {e}")

                    full_summary = "\n\n".join(summaries)
                    st.session_state.summary_content = full_summary
                    st.session_state.vectors = None  # Reset vectors
                        
                    status_text.empty()
                    st.success(f"✅ Website processed! {len(text.split()):,} words extracted.")
                    st.balloons()  # Celebration effect
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Some websites may block automated access")
    
else:
        # Getting started guide
    st.markdown("""
        ### 🎯 How to Use
        
        1. **Paste a URL** above (YouTube video or website)
        2. **Click the appropriate button** to process the content
        3. **Enable the chatbot** on the left to ask questions
        4. **Start chatting** about the summarized content
        
        **Supported Sources:**
        - 📺 YouTube videos with captions
        - 🌐 Websites and articles  
        - 📄 Blog posts and documentation
        """)
    
    # Display summary with enhanced styling
if st.session_state.summary_content:
    st.markdown("### 📝 Generated Summary")
    st.markdown(f'<div class="summary-response">{st.session_state.summary_content}</div>', unsafe_allow_html=True)

        # Action buttons
    col_clear, col_new = st.columns([1, 1])
    with col_clear:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.summary_content = ""
            st.session_state.final_text = ""
            st.session_state.chat_response = ""
            st.session_state.vectors = None                
            st.success("All data cleared!")
            st.rerun()
        
    with col_new:
        if st.button("🔄 Process New Content", use_container_width=True):
            st.session_state.summary_content = ""
            st.session_state.final_text = ""                
            st.session_state.chat_response = ""
            st.session_state.vectors = None
            st.rerun()
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 1rem; margin-top: 2rem;'>
        🚀 <strong>AI Content Assistant</strong> - Fixed, Enhanced, User-Friendly
    </div>
    """, 
    unsafe_allow_html=True
)
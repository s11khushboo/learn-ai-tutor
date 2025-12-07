# Joy – AI Tutor for After the Bootcamp

A Streamlit-powered learning assistant with vector search, YouTube/PDF ingestion, and AI reasoning.

Joy is an intelligent AI tutor built to help developers continue learning after finishing their coding bootcamp.
Bootcamps move fast. Once they’re done, many learners feel stuck. Joy solves this by letting you:

Upload your PDF notes,

Ingest and understand YouTube lectures,

Ask deep AI/ML/technical questions,

Search your personalized knowledge base using vectors,

Chat naturally through a clean Streamlit UI.

Joy becomes your ongoing mentor—patient, clear, and always available.


# 📦 Tech Stack
Component	Technology 

Interface	Streamlit

AI Model	ChatOpenAI / GPT

Transcription	OpenAI Whisper

Embeddings	all-MiniLM-L6-v2

Vector DB	Pinecone

Agent Framework	LangChain

PDF Processing	PyMuPDF (fitz) + Tesseract

OCR	pytesseract

Video Handling	yt_dlp


# Technical Architecture:

            ┌──────────────────────────┐
            │        Streamlit UI       │
            └─────────────┬────────────┘
                          ▼
                ┌──────────────────┐
                │ LangChain Agent  │
                └───────┬─────────┘
                
         ┌──────────────┼──────────────────────┐
         
         ▼              ▼                      ▼
         
 ingest_youtube  ingest_pdf_tool      search_vector_db
 
         │              │                      │
         
         ▼              ▼                      ▼
         
   Whisper ASR   PDF + OCR Engine        Pinecone Vector DB
   
         │              │
         └────── Embeddings ────────────┘
         
                          ▼
               **LangSmith Evaluation**
               
     (trace runs, monitor accuracy, debug tool calls)
     




# 🚀 Installation
1. Clone repository
   
  git clone https://github.com/yourname/joy-ai-tutor.git
  cd joy-ai-tutor


3. Install requirements
 
  pip install -r requirements.txt

5. Environment variables

  Create a .env file or use your keyloader system:
    
    OPENAI_API_KEY=your_key
    PINECONE_KEY=your_key

4. Run Streamlit UI
    streamlit run app.py


# 🧡 Joy’s Purpose

Bootcamp ends. Learning doesn’t.
Joy continues teaching you the things you didn’t have time to learn in class—
and the things you’re ready to understand now.

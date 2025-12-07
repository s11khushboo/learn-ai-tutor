from yt_dlp import YoutubeDL
import whisper
from sentence_transformers import SentenceTransformer
import math
import time
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import urlparse, parse_qs
from langchain_core.prompts import PromptTemplate  # pseudo imports
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import uuid
import fitz # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from urllib.parse import urlparse, unquote
import requests
from langchain.tools import tool
from keyloader import get_secret

INDEX_NAME = "ai-tutor"
EMBED_MODEL = "all-MiniLM-L6-v2"  # or OpenAI embeddings
WHISPER_MODEL = "small"

pc = Pinecone(api_key=get_secret("PINECONE_KEY"))

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

# embeddings (sentence-transformers)
embedder = SentenceTransformer(EMBED_MODEL)


def download_audio(youtube_url, out_path="audio.mp3"):
    ydl_opts = {"format": "bestaudio/best", "outtmpl": out_path}
    # download audio
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        info = ydl.extract_info(youtube_url, download=False)
        title = info.get("title", None)
    return out_path ,title

def get_video_id(url: str):
    # Extract video ID
    parsed = urlparse(url)
    if "youtu.be" in parsed.hostname:
        video_id = parsed.path[1:]
    elif "watch" in parsed.path:
        video_id = parse_qs(parsed.query)["v"][0]
    elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
        video_id = parsed.path.split("/")[2]
    else:
        raise ValueError("Unsupported YouTube URL format.")
    return video_id

# transcribe
def transcribe_whisper(audio_path):
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, task="transcribe")  # returns segments with timestamps
    return result

# chunking with overlap
def chunk_segments(segments, max_chars=1000, overlap_chars=200):
    chunks = []
    buffer = ""
    buffer_start = None
    buffer_end = None
    for seg in segments:
        text = seg["text"].strip()
        if not buffer:
            buffer_start = seg["start"]
        if len(buffer) + len(text) <= max_chars:
            buffer += (" " + text)
            buffer_end = seg["end"]
        else:
            chunks.append({
                "start": buffer_start, "end": buffer_end, "text": buffer.strip()
            })
            # start new buffer with overlap
            buffer = text[-overlap_chars:]
            buffer_start = seg["start"]
            buffer_end = seg["end"]
    if buffer:
        chunks.append({"start": buffer_start, "end": buffer_end, "text": buffer.strip()})
    return chunks




def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False).tolist()

# index_name = "ai-tutor"


# check if index already exists (it shouldn't if this is first time)
if INDEX_NAME not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(INDEX_NAME)
# view index stats
index.describe_index_stats()

def upsert_chunks(video_url,video_id, title, chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)  # this should be list of lists
    if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # ensure emb is a plain Python list
        if not isinstance(emb, list):
            emb = emb.tolist()

        # ensure metadata contains only serializable types
        metadata = {
            "source_type": "vedio",
            "content_type": "transcript",
            "chunk_id": i,
            "start_time": float(chunk["start"]),
            "end_time": float(chunk["end"]),
            "text": str(chunk["text"]),
            "source_name": str(title),
             "source_url": video_url,
        }


        vectors.append({
            "id": f"{video_id}_chunk_{i}",
            "values": emb,
            "metadata": metadata
        })

    # upsert all vectors
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} chunks for video {video_id}")

@tool
def ingest_youtube_video(url: str) -> str:
    """Ingest a YouTube video by downloading audio, transcribing, and storing chunks.

    Args:
        url: The YouTube video URL to ingest

    Returns:
        Success message with number of chunks ingested
    """
    print("Downloading audio...")
    audio_path, title = download_audio(url)
    transcript = transcribe_whisper(audio_path)
    chunks = chunk_segments(transcript["segments"])
    video_id = get_video_id(url)
    upsert_chunks(url,video_id, title, chunks)
    return f"Successfully ingested video: {url}. Chunks: {len(chunks)}"



def download_pdf(url):
    parsed = urlparse(url)

    # Extract last part of the path
    filename = os.path.basename(parsed.path)
    filename = unquote(filename)  # decode %20 etc.

    # If URL does not contain a real filename, generate one
    if not filename.lower().endswith(".pdf"):
        filename = "downloaded_file.pdf"
    folder = "./pdf"
    os.makedirs(folder, exist_ok=True)

    local_path = os.path.join(folder, filename)
    response = requests.get(url)
    with open(local_path, "wb") as f:
         f.write(response.content)
    return local_path

def ingest_pdf(pdf_path):
    print(pdf_path)
    doc = fitz.open(pdf_path)
    vectors = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )


    for page_num, page in enumerate(doc):

        # ---------- TEXT CHUNKS ----------
        page_text = page.get_text("text")

        if page_text.strip():
            chunks = splitter.split_text(page_text)

            for chunk_index, chunk_text in enumerate(chunks):

                metadata = {
                    "source_type": "pdf",
                    "source_name": pdf_path,
                    "page_number": page_num + 1,
                    "chunk_id": chunk_index,
                    "content_type": "text",
                    "text": chunk_text,
                    "source_url": pdf_path
                }

                vectors.append({
                    "id": f"{pdf_path}_p{page_num+1}_c{chunk_index}",
                    "values": embed_texts(chunk_text),
                    "metadata": metadata
                })


        # ---------- IMAGE OCR CHUNKS ----------
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            img_data = doc.extract_image(xref)["image"]
            img_obj = Image.open(io.BytesIO(img_data))

            ocr_text = pytesseract.image_to_string(img_obj)

            if ocr_text.strip():
                metadata = {
                    "source_type": "pdf",
                    "source_name": pdf_path,
                    "page_number": page_num + 1,
                    "image_index": img_index,
                    "chunk_id": f"img{img_index}",
                    "content_type": "image_text",
                    "text": ocr_text,
                    "source_url": pdf_path
                }

                vectors.append({
                    "id": f"{pdf_path}_p{page_num+1}_img{img_index}",
                    "values": embed_texts(ocr_text),
                    "metadata": metadata
                })

        # ---------- UPSERT TO PINECONE ----------
        if vectors:
            index.upsert(vectors=vectors)
            print(f"Inserted {len(vectors)} chunks into Pinecone from {pdf_path}.")

@tool
def ingest_pdf_tool(url: str) -> str:
    """Ingest a pdf by storing chunks.

    Args:
        url: pdf

    Returns:
        Success message with number of chunks ingested
    """
    local_url=download_pdf(url)
    print(local_url)
    ingest_pdf(local_url)

@tool
def search_vector_db(query: str) -> str:
    """Search the vector database for documents similar to the query."""

    # 1) Embed query
    q_emb = embedder.encode([query])[0]
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()

    # 2) Query Pinecone
    response = index.query(
        vector=q_emb,
        top_k=6,
        include_metadata=True
    )

    results = []

    for match in response.get("matches", []):
        meta = match.get("metadata", {})

        text_content = meta.get("text")
        if not text_content:
            text_content = meta.get("ocr_text")

        unified = {
            "score": match["score"],
            "source_type": meta.get("source_type"),
            "content_type": meta.get("content_type"),
            "source_name": meta.get("source_name"),
            "source_url": meta.get("source_url", None),
            "page_number": meta.get("page_number"),
            "image_index": meta.get("image_index"),
            "chunk_id": meta.get("chunk_id"),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "text": text_content
        }

        results.append(unified)

    print(results)
    print(f"Found {len(results)} results for query: {query}")

    # 3) Build context
    context = ""
    for md in results:

        if md["source_type"] == "video":
            context += (
                f"[{md['start_time']:.1f}s - {md['end_time']:.1f}s] "
                f"{md['text']}  (Source: {md['source_url']})\n\n"
            )
        else:
            # Handle PDFs (with or without URLs)
            src = md["source_url"] if md.get("source_url") else md.get("source_name")
            context += f"{md['text']}  (Source: {src})\n\n"

    print(context)
    return context

prompt="""You are an AI Knowledge Assistant that can answer  AI/ML/technical questions based on a knowledge base and manage a knowledge base of YouTube videos and PDFs.

You have three tools:
1. ingest_youtube_video(url)
2. ingest_pdf_tool(file_or_url)
3. search_vector_db(query)

Rules:

- Ingesting:
  • Call `ingest_youtube_video` for YouTube links.
  • Call `ingest_pdf_tool` for PDFs.


- Searching:
  • Call `search_vector_db .
  • When user ask questions, search the knowledge base..


- Behavior:
  • Be accurate and clear.
  • Do not hallucinate about documents.
  • If you dont find any document related to question then politely say i dont know
"""

def create_search_agent():
    """Create and return a configured search agent."""

    llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=get_secret("OPENAI_API_KEY"))
    checkpointer = MemorySaver()


    agent = create_agent(
        model=llm,
        tools=[ingest_youtube_video,ingest_pdf_tool,search_vector_db],
        system_prompt=prompt
    )


    return agent

agent = create_search_agent()

def answer_query(user_query):
   thread_id = str(uuid.uuid4())
   result=agent.invoke({
    "messages": [{"role": "user", "content":user_query }]
   },
   config={"configurable": {"thread_id": thread_id}} )
   answer = result["messages"][-1].content
   return answer

question = "what is Multi-Head Attention"
answer = answer_query(question)
print("Final Answer:")
print(answer)

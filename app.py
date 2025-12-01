import streamlit as st
import uuid
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time
import os
from dotenv import load_dotenv
import preprocessing_video as helper

load_dotenv()
EMBED_MODEL = "all-MiniLM-L6-v2"  # or OpenAI embeddings
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index_name = helper.INDEX_NAME
# connect to index
index = pc.Index(index_name)


# embeddings (sentence-transformers)
embedder = SentenceTransformer(EMBED_MODEL)
# Initialize session ID for memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.agent = helper.answer_query()

# UI

st.title("💬 AI Chat Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# --- SIDE PANEL CONTROLS ---
with st.sidebar:
    st.header("💾 Chat Controls")

    chat_history = st.session_state.get("messages", [])

    if chat_history:
        download_string = ""

        # Download Button
        st.download_button(
            label="Download Chat History (.txt)",
            data=download_string,
            file_name="streamlit_chatbot_history.txt",
            mime="text/plain",
            type="primary"
        )

        st.markdown("---")

        # Clear Button
        if st.button("Clear Chat", help="Permanently deletes all messages from memory."):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("Start chatting to enable download and clear controls!")
# Chat input
if user_input := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )

            response = result["messages"][-1].content
            st.write(response)

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response})
import streamlit as st

import preprocessing_video as helper
import uuid

# Setup
st.set_page_config(page_title="Chat Assistant", layout="wide")
st.title("🤖 AI Chat Assistant with Memory")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    # Create tools
    agent=helper.create_search_agent()

    st.session_state.agent = agent

# Sidebar
with st.sidebar:
    st.header("Settings")

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.write(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("⏳ Processing your request..."):
            try:
                # Invoke agent with memory
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                response = result["messages"][-1].content
                st.write(response)

                # Store assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
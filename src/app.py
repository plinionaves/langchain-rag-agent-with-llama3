from dotenv import load_dotenv
import streamlit as st
from agent import agent

load_dotenv()

with st.sidebar:
    collection_name = st.text_input("Vector DB collection name", "my-collection")
    collection_scope = st.text_input(
        "Collection scope (e.g. LLMs, Agents, Gen AI)", "LLMs, Agents, Gen AI"
    )
    urls = st.text_area(
        "URLs to be used as context (separated by new line)",
        "\n".join(
            [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            ]
        ),
        height=248,
    )

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by LLaMa3 LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    loading_message = {"role": "assistant", "content": "Generating response..."}
    st.session_state.messages.append(loading_message)
    st.chat_message("assistant").write(loading_message["content"])

    response = agent.generate_response(
        question=prompt,
        collection_name=collection_name,
        collection_scope=collection_scope,
        urls=urls.split("\n"),
    )

    st.session_state.messages.pop()
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

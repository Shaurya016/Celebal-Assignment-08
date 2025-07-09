# streamlit_app.py

import streamlit as st
from rag_utils import load_data, prepare_texts, build_vector_store, load_generator, ask_rag_bot, get_top_k_chunks

st.set_page_config(page_title="Loan Approval Q&A Chatbot")

st.title("💬 Loan Approval Q&A Chatbot")
st.markdown("Ask natural language questions about loan approvals using a smart RAG-based system.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def setup_rag():
    df = load_data()
    texts = prepare_texts(df)
    embed_model, index, _ = build_vector_store(texts)
    generator = load_generator()
    return df, texts, embed_model, index, generator

df, texts, embed_model, index, generator = setup_rag()

# Input box
query = st.text_input("🔍 Ask your question:", placeholder="e.g. How does education level affect loan approval?")

if query:
    with st.spinner("Generating intelligent response..."):
        answer = ask_rag_bot(query, embed_model, index, texts, generator)
        top_chunks = get_top_k_chunks(query, embed_model, index, texts)

    # Save to session state
    st.session_state.chat_history.append((query, answer))

    # Display result
    st.markdown("### 🧠 Answer:")
    st.write(answer)

    # Expandable section to show retrieved context
    with st.expander("📂 Retrieved context used by model"):
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")

# Display full chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🕓 Chat History")
    for q, a in st.session_state.chat_history[::-1]:  # newest first
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

try:
    answer = ask_rag_bot(query, embed_model, index, texts, generator)
    top_chunks = get_top_k_chunks(query, embed_model, index, texts)
except Exception as e:
    st.error(f"❌ An error occurred: {e}")


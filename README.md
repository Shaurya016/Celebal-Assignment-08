# Celebal-Assignment-08

# 💬 Loan Approval RAG Chatbot (Celebal Assignment)

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot that answers intelligent questions related to loan approvals. It uses document retrieval and a base transformer model for response generation, and is wrapped in an interactive **Streamlit** web application for a user-friendly interface.

---

## 🚀 Live Demo

👉 [Click here to open the chatbot](https://celebal-assignment-08-o378bdbaxt9wluencwcuse.streamlit.app/)

---

## 📚 Dataset

- [Loan Approval Dataset on Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)

---
## 📦 Tech Stack

| Component     | Technology                          |
|---------------|-------------------------------------|
| Interface     | Streamlit                           |
| Embeddings    | SentenceTransformers (`MiniLM`)     |
| Retrieval     | FAISS (vector similarity search)     |
| Generation    | Hugging Face (`flan-t5-base`)       |
| Dataset       | Kaggle - Loan Approval Dataset       |

---

## 📁 Project Structure

```
celebal-assignment-08/
│
├── rag_app.py              # Main Streamlit app
├── rag_utils.py            # Helper functions: RAG logic
├── Training Dataset.csv    # Dataset from Kaggle
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 💡 Features

✅ Ask natural questions like:
- "How does education level affect loan approval?"
- "Are self-employed people less likely to get approved?"
- "Give an example of a female applicant who got their loan."

✅ View:
- The generated answer
- Retrieved dataset rows as context
- Full chat history

## 🔧 Setup Instructions

1. **Clone the repo**
   ```
   git clone https://github.com/Shaurya016/Celebal-Assignment-08.git
   cd Celebal-Assignment-08
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```
   streamlit run rag_app.py
   ```

4. **(Optional) Deploy to Streamlit Cloud**
   - Create a free account at [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect your GitHub repo
   - Select `rag_app.py` as the main file
   - Deploy 🚀

---

## 🧪 Sample Questions

Here are a few sample queries you can ask the chatbot:

- "How does education level affect loan approval?"
- "Are married applicants more likely to get approved?"
- "Tell me about an applicant with income above 6000 who got approved."
- "What is the approval rate among self-employed people?"

---

---

## 📌 License

This project is for educational and academic use only.

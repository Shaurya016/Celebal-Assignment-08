# rag_utils.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

def load_data():
    df = pd.read_csv("Training Dataset.csv").dropna()
    return df

def row_to_text(row):
    return f"Applicant ID {row['Loan_ID']} is a {row['Gender']} who is {row['Married']} and is {row['Education']}. " \
           f"The applicant is {'self-employed' if row['Self_Employed'] == 'Yes' else 'not self-employed'}, " \
           f"has an income of {row['ApplicantIncome']}, requested a loan amount of {row['LoanAmount']} and " \
           f"their loan was {'approved' if row['Loan_Status'] == 'Y' else 'not approved'}."

def prepare_texts(df):
    return df.apply(row_to_text, axis=1).tolist()

def build_vector_store(texts):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(texts)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return embed_model, index, embeddings

def get_top_k_chunks(query, embed_model, index, texts, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [texts[i] for i in indices[0]]

def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def ask_rag_bot(query, embed_model, index, texts, generator):
    context = " ".join(get_top_k_chunks(query, embed_model, index, texts))
    prompt = f"Use the following data to answer:\n{context}\n\nQuestion: {query}"
    response = generator(prompt, max_length=150, do_sample=True)
    return response[0]['generated_text']

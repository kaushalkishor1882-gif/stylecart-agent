✅ FINAL CORRECTED README (Use this version)

Just replace yours with this 👇

# 🛍️ StyleCart AI Customer Support Agent

## 📌 Overview
The StyleCart AI Customer Support Agent is an intelligent chatbot built using **Agentic AI principles**. It helps users with common e-commerce queries such as returns, shipping, payments, exchanges, and order tracking.

The system uses **LangGraph** to create a multi-step reasoning workflow and ensures responses are **accurate, grounded, and reliable** using Retrieval-Augmented Generation (RAG).

---

## 🎯 Problem Statement
E-commerce platforms receive a large number of repetitive customer queries daily. Handling these manually is time-consuming and inefficient.

This project aims to build an AI-powered assistant that:
- Automates customer support  
- Provides accurate answers from company policies  
- Reduces workload on human agents  

---

## 💡 Solution
The solution is a **LangGraph-based AI agent** that:
- Retrieves relevant information from a knowledge base (ChromaDB)  
- Uses an LLM (Groq) to generate responses  
- Maintains conversation memory  
- Uses tools (date/time) when needed  
- Evaluates its own answers to prevent hallucination  

---

## ⚙️ Features

- 🔍 **RAG (Retrieval-Augmented Generation)** using ChromaDB  
- 🧠 **LangGraph Workflow** with multiple nodes  
- 💬 **Conversation Memory** using thread_id  
- 🛠️ **Tool Integration** (Current Date & Time)  
- ✅ **Self-Evaluation Node** to ensure faithfulness  
- 🖥️ **Streamlit UI** for interactive chat  
- 🚫 **No Hallucination Policy** (Strict grounding to context)  

---

## 🏗️ Architecture


User Input
↓
Memory Node → Router Node
↓
Retrieve / Tool / Skip
↓
Answer Node
↓
Evaluation Node (Faithfulness Check)
↓
Save Node → Response


---

## 🧰 Tech Stack

- **Programming Language:** Python  
- **LLM:** Groq (LLaMA 3.3 70B)  
- **Framework:** LangGraph  
- **Vector Database:** ChromaDB  
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)  
- **Frontend:** Streamlit  

---

## 📂 Project Structure


StyleCart_Project/
├── agent.py
├── capstone_streamlit.py
├── part1.py
├── part2_3.py
├── part4_5.py
├── part6.py
├── README.md


---

## ▶️ How to Run

### 1. Install Dependencies

pip install -r requirements.txt


### 2. Set Groq API Key

set GROQ_API_KEY=your_api_key_here


### 3. Run Streamlit App

streamlit run capstone_streamlit.py


---

## 🧪 Testing

The project includes:
- ✅ 10 domain-specific test cases  
- ✅ Red-team tests (out-of-scope & prompt injection)  
- ✅ Memory test (multi-turn conversation)  

---

## 📊 Evaluation

The system is evaluated using:
- **Faithfulness**  
- **Answer Relevancy**  
- **Context Precision**  

(RAGAS framework used for evaluation)

---

## ⭐ Unique Features

- 🔁 Self-correcting AI (retry on low faithfulness)  
- 🧠 Memory-aware conversations  
- 🔒 Strict grounding (no external knowledge)  
- 🏬 Real-world e-commerce use case  

---

## 🔮 Future Improvements

- 🌐 Multilingual support  
- 📱 WhatsApp integration  
- 📦 Real-time order tracking API  
- 🎤 Voice-based assistant  

---

## 👤 Author

**Kaushal Kishor**  
Roll Number: **23051597**  
Agentic AI Capstone Project – 2026  

---

## 📌 Note
This project is developed as part of the **Agentic AI Capstone Project** and demonstrates practical implementation of LLM-based intelligent agents using modern AI frameworks.

# Serverless RAG Q&A Application 🚀

Welcome to the Serverless RAG (Retrieval-Augmented Generation) Q&A Application! This Streamlit-based web app allows you to upload various documents (PDFs, Text Files, DOCX, or web Links) and ask questions about their content.

Under the hood, it uses **Hugging Face Serverless Inference** to power a Meta Llama 3 conversational model and uses Sentence Transformers with a local FAISS vector database to grab the most relevant context for your questions. All without needing an expensive GPU or paid OpenAI API keys!

## ✨ Features
- **Multi-Format Support:** Easily process web URLs, plain Text, PDFs, and Word documents (`.docx`).
- **Free-Tier Friendly:** Leverages the Hugging Face Serverless API. 
- **Local Vectors:** Uses `faiss-cpu` and `sentence-transformers` to locally chunk and store vectors, preventing data leaks and keeping memory usage clean and quick.
- **Mac Optimization:** Built-in safeguards automatically handle Apple Silicon M1/M2/M3 specific multiprocessing quirks!

## 🛠️ Setup & Installation

### 1. Clone the repository
Make sure you have downloaded or cloned this codebase to your machine and open your terminal in the project folder.

### 2. Install Dependencies
It is highly recommended to create a virtual environment first. Once your environment is active, install the required packages using the newly provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Get your Hugging Face API Key
To use the Meta Llama 3 model for generating answers, you need a free Hugging Face API Key.
1. Sign up or log into [Hugging Face](https://huggingface.co).
2. Go to **Settings** -> **Access Tokens** (https://huggingface.co/settings/tokens).
3. Create a new token.
4. **CRITICAL:** When creating the token, make sure you explicitly check the box under Permissions that says **"Make calls to the Serverless Inference API"**. This app will not work without it! (If you get a 403 Forbidden error, this is usually why).

### 4. Run the App
Start the Streamlit application by running the following command in your terminal:

```bash
streamlit run app.py
```

This will automatically open the app in your default web browser (usually at `http://localhost:8501`).

## 📖 How to Use
1. Locate the sidebar on the left side of the web page.
2. Enter your **Hugging Face API Key**.
3. Select your **Input Type** (e.g., Link, PDF).
4. Enter the URL(s) or upload your document(s) in the main panel.
5. Click **Proceed**. The app will chunk your documents and build a temporary FAISS vector database in your session. 
6. Once you see the success message, type your question in the text box at the bottom and click **Submit**.
7. Enjoy your AI-generated answers based on your documents!

## ⚠️ Common Issues
- **`403 Forbidden Error`**: Your Hugging Face token lacks the necessary permissions. Go back to your Hugging Face settings and create a new token with the Inference permissions explicitly enabled.
- **`Segmentation Fault`**: If you see this on a Mac, make sure you did not manually alter the top of `app.py`. The environment variables defined there (`TOKENIZERS_PARALLELISM`, etc.) must run *before* Torch or FAISS are imported.

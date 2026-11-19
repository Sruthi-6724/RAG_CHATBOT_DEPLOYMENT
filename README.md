# NovaTech Internal Assistant – RAG Chatbot

## Project Overview
NovaTech Internal Assistant is an AI-powered chatbot that helps employees find information about company policies, products, and procedures. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from internal documents and generate accurate, context-based responses.

## Features
- Answers questions about company policies, products, and procedures.
- Retrieves relevant information from internal text documents.
- Generates context-based responses using Groq LLM.
- Displays source documents for generated answers.
- Maintains conversation history.
- Supports two document retrieval approaches: ChromaDB and TF-IDF.

## Technologies Used
- Python
- Streamlit
- Groq API
- OpenAI Python SDK
- ChromaDB
- TF-IDF
- Scikit-learn
- Python-dotenv

## Project Structure

```text
NOVATECH-RAG-CHATBOT/
│
├── main.py
├── streamlit_app.py
├── streamlit_app_tfidf.py
├── requirements.txt
├── .env
│
├── _data/
│   ├── company_policy.txt
│   ├── products.txt
│   └── procedures.txt
│
└── venv/
```

## RAG Approaches

### 1. ChromaDB-Based RAG
- Splits documents into paragraphs.
- Stores document chunks in ChromaDB.
- Uses semantic similarity search to retrieve relevant chunks.
- Sends retrieved context to the Groq LLM.

### 2. TF-IDF-Based RAG
- Splits documents into paragraphs.
- Converts text into TF-IDF vectors.
- Uses cosine similarity to retrieve relevant chunks.
- Sends retrieved context to the Groq LLM.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd NOVATECH-RAG-CHATBOT
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the Application

ChromaDB-based application:

```bash
streamlit run streamlit_app.py
```

TF-IDF-based application:

```bash
streamlit run streamlit_app_tfidf.py
```

## How It Works

1. User enters a question.
2. The system searches internal documents.
3. Relevant document chunks are retrieved.
4. Retrieved information is added to the LLM prompt.
5. Groq LLM generates a context-based response.
6. The chatbot displays the answer along with its source documents.

## Security
- API keys are stored in environment variables.
- The `.env` file should not be committed to GitHub.
- Internal company documents should only be shared with authorized users.

## Future Enhancements
- Support for PDF and DOCX documents.
- Improved document chunking strategies.
- User authentication.
- More advanced retrieval techniques.
- Deployment to a cloud platform.

## Author
Developed as a Python-based Retrieval-Augmented Generation (RAG) chatbot project.
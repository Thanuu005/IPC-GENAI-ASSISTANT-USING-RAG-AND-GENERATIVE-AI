⚖️ IPC GenAI Assistant using RAG and Generative AI

An educational AI-powered application that answers questions about an
IPC (Indian Penal Code) dataset. It uses semantic search to retrieve
relevant dataset records and Google Gemini to generate a simple,
readable response. The application interface is built with Streamlit.

Disclaimer: This project is for educational and informational
purposes only. It is not legal advice. The IPC was replaced by the
Bharatiya Nyaya Sanhita (BNS) for most new offences from 1 July 2024.
Always verify current law using authoritative sources.

Features

Ask questions about records in an IPC CSV dataset.

Convert dataset records into embeddings using Sentence Transformers.

Retrieve relevant records with FAISS vector search.

Generate responses using the Google Gemini API.

Display retrieved dataset records to help users inspect the
information used.

Run locally or deploy as a Streamlit web application.

Technology Stack

Python

Streamlit

Pandas and NumPy

Sentence Transformers (all-MiniLM-L6-v2)

FAISS (faiss-cpu)

Google Gen AI SDK (google-genai)

Google Gemini API

Project Structure

IPC-GENAI-ASSISTANT/
├── .streamlit/
│   └── secrets.toml       # Local secrets file; do not commit
├── app.py                 # Streamlit application
├── ipc_dataset.csv        # IPC dataset (check redistribution permissions)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md

Requirements

Python 3.10 or a compatible Python version supported by the
installed libraries

Internet access for downloading the embedding model and calling
Gemini

A Google Gemini API key

Your IPC dataset in CSV format

Setup and Run Locally

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/IPC-GenAI-Assistant.git
cd IPC-GenAI-Assistant

Replace YOUR-USERNAME with your GitHub username and adjust the
repository name if needed.

2. Create and activate a virtual environment

On Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

If PowerShell blocks activation, you can run the environment's Python
directly:

.\venv\Scripts\python.exe -m pip install -r requirements.txt

3. Install dependencies

python -m pip install -r requirements.txt

4. Add your Gemini API key

Create the folder .streamlit and inside it create secrets.toml:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

Replace the placeholder with your own key. Never commit secrets.toml,
.env, or your API key to GitHub.

The application reads the key using Streamlit secrets, for example:

api_key = st.secrets.get("GEMINI_API_KEY")

5. Add the dataset

Place your permitted IPC CSV dataset in the project root and name it:

ipc_dataset.csv

If your file has a different name, update the dataset filename in
app.py. The application builds searchable text records from the CSV
columns, so answer quality depends on the dataset's content and
structure.

6. Start Streamlit

python -m streamlit run app.py

Open the local address shown in the terminal, commonly
http://localhost:8501.

Deploy on Streamlit Community Cloud

Push the project files to a GitHub repository.

Open Streamlit Community Cloud.

Create a new app and select your repository, branch, and app.py as
the main file.

In the app's Secrets settings, add:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

Deploy the app and check the logs if the build fails.

Do not upload .streamlit/secrets.toml, API keys, virtual
environments, or private data to GitHub. Only publish the dataset if you
have permission to redistribute it.

How It Works

Load data: Pandas reads and cleans the IPC CSV file.

Create documents: Each dataset row is converted into searchable
text.

Create embeddings: Sentence Transformers converts the text into
vectors.

Build search index: FAISS indexes the vectors for similarity
search.

Retrieve context: The user's question is embedded and the most
similar records are retrieved.

Generate answer: The retrieved records are passed to Gemini with
instructions to answer cautiously.

Show results: Streamlit displays the answer and retrieved
dataset records.

This approach is called Retrieval-Augmented Generation (RAG).

Troubleshooting

Import "faiss" could not be resolved: Install dependencies in
the same Python environment selected by VS Code:
python -m pip install faiss-cpu.

Import "sentence_transformers" could not be resolved: Run
python -m pip install sentence-transformers in the selected
environment.

Missing API key: Check local .streamlit/secrets.toml or the
deployed app's Streamlit Secrets settings.

Model unavailable: Update the Gemini model name in app.py to a
model currently available to your API key.

Slow first startup: The embedding model must download, and the
application creates embeddings and a FAISS index. Startup may take
time depending on dataset size and hosting resources.

Answers say the information is unavailable: Check the CSV
columns and sample rows, then confirm the retrieved records contain
the relevant section. The model cannot reliably answer facts missing
from the dataset.

Limitations and Responsible Use

Answers depend on the uploaded dataset and retrieved records;
missing or inaccurate source data can lead to incomplete answers.

Generative AI can make mistakes. Verify legal information
independently.

This project is an educational reference tool, not a substitute for
a lawyer or an authoritative legal source.

Do not upload confidential or personal information to a public
repository.

Protect the Gemini API key and monitor API usage.

Future Improvements

Improve retrieval with section-aware search and metadata filters.

Add evaluation questions and retrieval-quality measurements.

Persist precomputed embeddings/indexes to reduce startup time.

Add dataset validation and clearer source citations.

Add tests and improve error handling.

Author

Project: IPC GenAI Assistant using RAG and Generative AI
Purpose: Educational project demonstrating retrieval-augmented
question answering over a legal-reference dataset.
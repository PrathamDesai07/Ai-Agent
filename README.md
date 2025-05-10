# RAG-Powered Multi-Agent Q&A Assistant

This project is a Retrieval-Augmented Generation (RAG) powered Q&A assistant built with Flask. It allows users to upload PDF documents, create a custom knowledge base, and ask questions about the content. Additionally, it features a weather agent that can answer weather-related queries for any city.

## Features

- **Document Q&A (RAG):**
  - Upload a PDF to create a knowledge base.
  - Ask questions about the uploaded document.
  - Uses vector search (ChromaDB) and Google Generative AI embeddings for context retrieval.
  - Answers are generated using a language model based on the most relevant document chunks.

- **Weather Agent:**
  - Recognizes weather-related queries (e.g., "What's the weather in Paris?").
  - Fetches real-time weather data using the Weatherstack API.
  - Returns current weather description, temperature, and feels-like temperature for the specified city.

- **Chit-Chat Handling:**
  - Recognizes greetings and responds in a friendly manner.
  - Handles empty or irrelevant queries gracefully.

## How It Works

1. **Upload a PDF:**
   - Go to `/new-bot` and upload your PDF document.
   - The system processes the PDF, creates a vector database, and initializes the retriever.

2. **Ask Questions:**
   - On the main page (`/`), enter your question.
   - If the question is about the document, the RAG pipeline retrieves relevant context and generates an answer.
   - If the question is about the weather, the weather agent fetches and returns the current weather for the specified city.
   - If you greet the assistant (e.g., "hi"), it responds with a friendly message.

## Project Structure

- `app.py` — Main Flask application, routing, and agent logic.
- `RAG/` — Contains RAG pipeline code, vector DB processing, and related utilities.
- `document/` — Folder for uploaded PDF files.
- `templates/` — HTML templates for the web interface.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd Ai-Agent
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Keys:**
   - **Google Generative AI:** Set `GOOGLE_API_KEY` as an environment variable or edit in `app.py`.
   - **Weatherstack:** The API key is set in `app.py` (replace with your own for production).

4. **Run the App:**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000` (or the port specified in `app.py`).

## Usage

- **Create a New Bot:**
  - Visit `/new-bot` and upload a PDF to initialize the knowledge base.
- **Ask Questions:**
  - Go to `/` and enter your question.
  - Try questions like:
    - "What is the main topic of the document?"
    - "Summarize section 2."
    - "What's the weather in New York?"
    - "Hi"

## Customization

- **Change Vector DB or Embeddings:**
  - Modify the RAG pipeline in `RAG/` as needed.
- **Add More Agents:**
  - Extend the Flask routes and logic in `app.py` to add more specialized agents.

## License

MIT License. See `LICENSE` file for details.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [Weatherstack API](https://weatherstack.com/)
- [Google Generative AI](https://ai.google/)

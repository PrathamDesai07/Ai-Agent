from flask import Flask, request, render_template, redirect, url_for, flash
import os
import shutil
import requests
import re
# --- RAG imports ---
from RAG.RAG import run_rag
from RAG.vector_db import process_pdf_to_vector_db
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
UPLOAD_FOLDER = './document/'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

chroma_db_dir = "./RAG/chroma_db_"
api_key = None
k = 5
retriever = None

def init_retriever():
    global retriever
    if os.path.exists(chroma_db_dir) and os.path.isdir(chroma_db_dir):
        if api_key is None:
            key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDlVCKsmkbHbQHl49zHkzbBbQ7iTRmdBSM')
        else:
            key = api_key
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/embedding-001")
        db_connection = Chroma(persist_directory=chroma_db_dir, embedding_function=embedding_model)
        retriever = db_connection.as_retriever(search_kwargs={"k": k})
    else:
        retriever = None

init_retriever()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Chatbot (Q&A) route at root ---
@app.route('/', methods=['GET', 'POST'])
def qa():
    answer = None
    question = None
    if request.method == 'POST':
        question = request.form.get('question')
        if not retriever:
            answer = "Retriever not initialized. Please create a new RAG bot by uploading a PDF."
        elif question:
            # Chit-chat/greeting detection
            greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
            if question.strip().lower() in greetings:
                answer = "Hello! How can I assist you with your document today?"
            else:
                # Weather query classification
                weather_pattern = re.compile(r"what(?:'s| is)? the weather (?:today|now)?(?: at| in)? ([a-zA-Z\s]+)", re.IGNORECASE)
                match = weather_pattern.search(question)
                if match:
                    city = match.group(1).strip()
                    api_key = "104af4b497fb3d0fe2180abadf2f25ea"
                    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
                    try:
                        resp = requests.get(url)
                        data = resp.json()
                        if 'current' in data:
                            weather_desc = data['current']['weather_descriptions'][0]
                            temp = data['current']['temperature']
                            feelslike = data['current']['feelslike']
                            answer = f"The weather in {city} is {weather_desc} with a temperature of {temp}°C (feels like {feelslike}°C)."
                        elif 'error' in data:
                            answer = f"Weather API error: {data['error'].get('info', 'Unknown error')}"
                        else:
                            answer = "Could not retrieve weather information."
                    except Exception as e:
                        answer = f"Error fetching weather: {e}"
                else:
                    # RAG pipeline
                    rag_answer = run_rag(question, retriever=retriever)
                    if not rag_answer or rag_answer.strip() == "":
                        answer = "I'm sorry, I couldn't find relevant information in the document."
                    else:
                        answer = rag_answer
    return render_template('qa.html', answer=answer, question=question)

# --- New RAG bot creation route ---
@app.route('/new-bot', methods=['GET', 'POST'])
def new_bot():
    if request.method == 'POST':
        # Delete old PDF(s)
        for fname in os.listdir(UPLOAD_FOLDER):
            if fname.lower().endswith('.pdf'):
                os.remove(os.path.join(UPLOAD_FOLDER, fname))
        # Delete old chroma_db_
        if os.path.exists(chroma_db_dir):
            shutil.rmtree(chroma_db_dir)
        # Save new PDF
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Allowed file type is pdf')
            return redirect(request.url)
        filename = file.filename
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        # Build new vector DB
        process_pdf_to_vector_db(pdf_path=pdf_path, chroma_db_dir=chroma_db_dir)
        # Re-initialize retriever
        init_retriever()
        # Set permissions
        os.chmod(chroma_db_dir, 0o777)
        for root, dirs, files in os.walk(chroma_db_dir):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o666)
        flash('New RAG bot created!')
        return redirect(url_for('qa'))
    return render_template('upload.html', new_bot=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
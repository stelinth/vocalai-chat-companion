import streamlit as st
import pyttsx3
import speech_recognition as sr
import spacy
import sqlite3
import threading
from hashlib import sha256
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import pdfplumber
import os

# Set your API keys here
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("Groq API key not set in environment variables.")
    st.stop()  # Stop further execution if the API key is not set

# Initialize spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Initialize text-to-speech engine
tts_engine = None
tts_lock = threading.Lock()
stop_tts = threading.Event()  # Event to signal stop TTS

# Database connection
def get_db_connection():
    return sqlite3.connect('user_data.db', check_same_thread=False)

def create_tables(conn):
    with conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                feedback TEXT,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        ''')

# Initialize conversation chain, memory, and session state
if 'conversation' not in st.session_state:
    # Initialize memory for conversation
    conversational_memory_length = 5  # Adjust as needed
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='mixtral-8x7b-32768'  # Default model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    st.session_state.conversation = conversation

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'feedback' not in st.session_state:
    st.session_state.feedback = []

# Function to hash passwords
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()
    return tts_engine

def text_to_speech(text):
    engine = get_tts_engine()
    with tts_lock:
        stop_tts.clear()  # Clear any previous stop signals
        try:
            sentences = text.split('.')
            for sentence in sentences:
                if sentence.strip():  # Check if sentence is not empty
                    engine.say(sentence)
            engine.runAndWait()
        except RuntimeError as e:
            if str(e) != 'run loop already started':
                raise e

def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening for your question...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except AttributeError:
        st.error("PyAudio is not installed. Please install PyAudio to use the microphone feature.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return None

def register_user(conn, username, password):
    password_hash = hash_password(password)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        st.success(f"User '{username}' registered successfully!")
    except sqlite3.IntegrityError:
        st.error(f"Username '{username}' already exists. Please choose a different username.")

def login_user(conn, username, password):
    password_hash = hash_password(password)
    with conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', (username, password_hash))
        user = cursor.fetchone()
    if user:
        st.success(f"Welcome back, {username}!")
        st.session_state.logged_in = True
        st.session_state.username = username
    else:
        st.error("Incorrect username or password.")
        st.session_state.logged_in = False

def preprocess_input(user_input):
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    sentiment = doc.sentiment
    return {'entities': entities, 'pos_tags': pos_tags, 'sentiment': sentiment}

def save_feedback(conn, username, feedback):
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO feedback (username, feedback) VALUES (?, ?)', (username, feedback))

        feedback_file_path = os.path.join('C:\\Users\\Stalin\\Downloads', 'feedback.txt')
        with open(feedback_file_path, 'a') as file:
            file.write(f'Username: {username}\nFeedback: {feedback}\n\n')
        st.success("Thank you for your feedback!")
    except Exception as e:
        st.error(f"An error occurred while saving feedback: {e}")

def save_conversation(conn, username, user_message, bot_response):
    with conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO conversations (username, user_message, bot_response) VALUES (?, ?, ?)',
                       (username, user_message, bot_response))

def get_user_conversations(conn, username):
    with conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, user_message, bot_response, timestamp FROM conversations WHERE username = ? ORDER BY timestamp DESC', (username,))
        return cursor.fetchall()

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

def main():
    st.title("VocalAI Chat Companion")
    st.subheader("Interactive Voice and Text Chatbot")

    # Initialize database connection
    conn = get_db_connection()
    create_tables(conn)

    # Initialize conversation object from session state
    conversation = st.session_state.conversation

    # Initialize logged-in state and chat history
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        # Authentication sidebar
        st.sidebar.title("User Authentication")
        auth_choice = st.sidebar.radio("Choose an option", ("Login", "Register"))

        # Handle registration
        if auth_choice == "Register":
            new_username = st.sidebar.text_input("Enter new username")
            new_password = st.sidebar.text_input("Enter new password", type="password")

            if st.sidebar.button("Register"):
                register_user(conn, new_username, new_password)

        # Handle login
        elif auth_choice == "Login":
            username = st.sidebar.text_input("Enter username")
            password = st.sidebar.text_input("Enter password", type="password")

            if st.sidebar.button("Login"):
                login_user(conn, username, password)

    # Display previous conversations for logged-in users
    if st.session_state.logged_in:
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.experimental_rerun()

        # Voice command activation
        if st.button("Talk"):
            user_question = recognize_speech_from_microphone()
            if user_question:
                preprocessed_input = preprocess_input(user_question)
                response = conversation(user_question)
                st.session_state.chat_history.append({'human': user_question, 'AI': response['response']})
                st.write("Chatbot:", response['response'])
                text_to_speech(response['response'])
                save_conversation(conn, st.session_state.username, user_question, response['response'])

        # Text input for user question
        user_question = st.text_input("Type your question:", key="user_question")

        if user_question:
            if st.button("Send"):
                preprocessed_input = preprocess_input(user_question)
                response = conversation(user_question)
                st.session_state.chat_history.append({'human': user_question, 'AI': response['response']})
                st.write("Chatbot:", response['response'])
                save_conversation(conn, st.session_state.username, user_question, response['response'])

        # Upload PDF for text extraction and analysis
        st.sidebar.title("Upload PDF for Analysis")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            st.sidebar.write(f"Uploaded File: {uploaded_file.name}")

            # Extract text from uploaded PDF and analyze it
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()

            st.write("Text extracted from PDF:")
            st.write(full_text)

            # Analyze extracted text using the chatbot
            response = conversation(full_text)
            st.write("Chatbot Analysis:")
            st.write(response['response'])

            # Save conversation to database
            save_conversation(conn, st.session_state.username, f"Uploaded PDF: {uploaded_file.name}", response['response'])

    # Provide feedback section
    st.sidebar.title("Provide Feedback")
    feedback_text = st.sidebar.text_area("Share your feedback:", height=100)
    if st.sidebar.button("Submit Feedback"):
        if feedback_text.strip():
            save_feedback(conn, st.session_state.username, feedback_text)
        else:
            st.warning("Feedback cannot be empty. Please provide your feedback.")

    # Display chat history
    st.header("Chat History")
    for chat in st.session_state.chat_history:
        st.text_area(f"{chat['human']}:", chat['AI'], height=100, key=f"{chat['human']}_chat")

    # Close database connection
    conn.close()

if __name__ == "__main__":
    main()

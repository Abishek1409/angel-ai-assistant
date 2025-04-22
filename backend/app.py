from flask import Flask, request, jsonify
import cohere
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

# Enable CORS for your Flask app
CORS(app)

if not os.getenv('COHERE_API_KEY'):
    raise EnvironmentError("❌ Error: COHERE_API_KEY not found in .env file. Please add your API key to the .env file.")

co = cohere.Client(os.getenv('COHERE_API_KEY'))

def get_answer(question):
    try:
        # Generate response using Cohere
        response = co.chat(
            message=question,
            max_tokens=200,
            temperature=0.7,
        )
        if response and response.text:
            return response.text.strip()
        return "I'm sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        return f"Error getting answer: {str(e)}"

def summarize_text(text):
    try:
        # Generate summary using Cohere
        response = co.summarize(
            text=text,
            length='medium',
            format='paragraph',
            extractiveness='medium',
            temperature=0.3,
        )
        if response and response.summary:
            return response.summary
        return "Error: Could not generate summary. Please try again."
    except Exception as e:
        return f"Error in summarization: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    if not question:
        return jsonify({"error": "Message is required"}), 400
    response = get_answer(question)
    return jsonify({"response": response})

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    summary = summarize_text(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from src.llm.question_generator import generate_clarification_questions

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json.get('feedback')
    questions = generate_clarification_questions(feedback)
    return jsonify(questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
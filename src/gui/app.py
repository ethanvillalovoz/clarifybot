from flask import Flask, render_template, request, jsonify
from llm.question_generator import QuestionGenerator

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

question_generator = QuestionGenerator(llm_model=None)  # Replace None with your actual model if available

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json.get('feedback')
    questions = question_generator.generate_clarification_questions(feedback)
    return jsonify(questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
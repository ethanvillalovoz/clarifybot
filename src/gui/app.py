from flask import Flask, render_template, request, jsonify
from llm.question_generator import QuestionGenerator
from rl.reward_model import RewardModel

app = Flask(__name__)

reward_model = RewardModel()
question_generator = QuestionGenerator(llm_model=None)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    conversation = data.get('conversation', [])
    feedback = data.get('feedback', '')

    questions = question_generator.generate_clarification_questions(feedback)
    summary = None
    if len(conversation) >= 6:
        summary = reward_model.summarize_preferences(conversation)

    return jsonify(questions=questions, summary=summary)

def run_gui(reward_model, question_generator):
    app.run(debug=True)
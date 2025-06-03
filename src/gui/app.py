from flask import Flask, render_template, request, jsonify
from llm.question_generator import QuestionGenerator
from rl.reward_model import RewardModel

# Add these imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from difflib import SequenceMatcher

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load Mistral model and tokenizer
mistral_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3", use_auth_token=True
)
mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3", use_auth_token=True
).to(device)

llm_model = {"model": mistral_model, "tokenizer": mistral_tokenizer, "device": device}

app = Flask(__name__)

reward_model = RewardModel()
question_generator = QuestionGenerator(llm_model=llm_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    conversation = data.get('conversation', [])
    feedback = data.get('feedback', '')
    language = data.get('language', 'English')
    questions = question_generator.generate_clarification_questions(feedback, conversation, language)
    summary = None
    if len(conversation) >= 6:
        summary = reward_model.summarize_preferences(conversation, llm_model=llm_model)

    # --- New uncertainty/misalignment logic ---
    uncertainty = compute_uncertainty(questions)
    confidence = round((1 - uncertainty) * 100)
    misalignment_flag = confidence < 20  # Only warn if confidence is below 60%

    if misalignment_flag:
        with open("misalignment_events.log", "a") as f:
            f.write(f"UNCERTAIN: {uncertainty:.2f}\t{feedback}\t{questions}\n")

    return jsonify(
        questions=questions,
        summary=summary,
        confidence=confidence,
        misalignment=bool(misalignment_flag)  # <-- Make sure this is a Python bool
    )

@app.route('/rate_question', methods=['POST'])
def rate_question():
    data = request.json
    question = data.get('question')
    rating = data.get('rating')  # 'up' or 'down'
    # Save to a log file for now
    with open("question_ratings.log", "a") as f:
        f.write(f"{question}\t{rating}\n")
    return jsonify(success=True)

def compute_uncertainty(questions):
    """
    Simple uncertainty metric: higher if questions are more diverse.
    """
    if not questions or len(questions) == 1:
        return 0  # Low uncertainty if only one question

    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    similarities = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            similarities.append(similarity(questions[i], questions[j]))

    avg_similarity = np.mean(similarities) if similarities else 1.0
    uncertainty = 1 - avg_similarity  # Higher = more diverse = more uncertain
    return uncertainty

def run_gui():
    app.run(debug=True)
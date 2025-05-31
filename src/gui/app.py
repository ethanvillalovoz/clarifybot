from flask import Flask, render_template, request, jsonify
from llm.question_generator import QuestionGenerator
from rl.reward_model import RewardModel

# Add these imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

    questions = question_generator.generate_clarification_questions(feedback)
    summary = None
    if len(conversation) >= 6:
        summary = reward_model.summarize_preferences(conversation)

    return jsonify(questions=questions, summary=summary)

def run_gui():
    app.run(debug=True)
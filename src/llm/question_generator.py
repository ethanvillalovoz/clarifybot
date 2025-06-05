from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rapidfuzz import fuzz
import random

PROMPT_TEMPLATES = [
    (
        "You are a helpful, empathetic assistant. "
        "Given the following conversation and user feedback, generate 3 open-ended, non-repetitive, and actionable clarification questions. "
        "Reference specific details from the user's feedback and avoid generic or previously asked questions. "
        "Each question should address a different aspect of the user's situation (for example: emotions, practical details, or future goals). "
        "Avoid yes/no questions, be supportive, and do not repeat yourself or previous questions in this conversation.\n"
        "User feedback: \"{feedback}\"\n\nQuestions:\n1."
    ),
    (
        "Imagine you are supporting someone who just shared the following feedback. "
        "Write 3 unique, open-ended questions to help them clarify their thoughts and feelings. "
        "Be gentle, supportive, and specific. Reference details from their feedback and avoid generic questions.\n"
        "User feedback: \"{feedback}\"\n\nQuestions:\n1."
    )
]

EPSILON = 0.1  # Exploration rate (10%)

def select_template(reward_model):
    """
    Epsilon-greedy selection for prompt templates using Bayesian mean.
    With probability EPSILON, select a random template (exploration).
    Otherwise, select the template with the highest posterior mean (exploitation).
    """
    if random.random() < EPSILON:
        idx = random.randint(0, len(PROMPT_TEMPLATES) - 1)
        mode = "explore"
    else:
        means = [reward_model.posterior_mean(i) for i in range(len(PROMPT_TEMPLATES))]
        max_mean = max(means)
        best_idxs = [i for i, m in enumerate(means) if m == max_mean]
        idx = random.choice(best_idxs)
        mode = "exploit"
    return idx, PROMPT_TEMPLATES[idx], mode

def load_bad_questions(log_path="question_ratings.log", threshold=3):
    from collections import Counter
    counter = Counter()
    try:
        with open(log_path) as f:
            for line in f:
                q, rating = line.strip().split('\t')
                if rating == 'down':
                    counter[q] += 1
    except FileNotFoundError:
        return set()
    return set(q for q, count in counter.items() if count >= threshold)

BAD_QUESTIONS = load_bad_questions()

class QuestionGenerator:
    def __init__(self, llm_model=None, reward_model=None):
        if llm_model is not None:
            self.model = llm_model["model"]
            self.tokenizer = llm_model["tokenizer"]
            self.device = llm_model["device"]
        else:
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
        self.reward_model = reward_model

    def generate_clarification_questions(self, feedback, conversation=None, language="English"):
        idx, prompt_template, mode = select_template(self.reward_model)
        prompt = prompt_template.format(feedback=feedback)

        context = ""
        if conversation:
            # Use the last 5 exchanges for more context
            context = "Conversation so far:\n" + "\n".join(
                f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation[-5:]
            ) + "\n"

        # Diverse example blocks for priming
        example_blocks = [
            "",
            (
                "Example questions:\n"
                "1. What has been weighing on your mind lately?\n"
                "2. Is there something specific you wish would change?\n"
                "3. What would make you feel more supported right now?\n"
            ),
            (
                "Example questions:\n"
                "1. Can you describe a recent moment that made you feel this way?\n"
                "2. What are some things that usually help you cope?\n"
                "3. Are there any goals or dreams you want to pursue?\n"
            ),
            (
                "Example questions:\n"
                "1. What specific event or thought triggered these feelings?\n"
                "2. Who do you usually turn to for support in times like this?\n"
                "3. What would you like to accomplish or change in the near future?\n"
            )
        ]
        example_block = random.choice(example_blocks)

        prompt_templates = [
            (
                "You are a helpful, empathetic assistant. "
                "Given the following conversation and user feedback, generate 3 open-ended, non-repetitive, and actionable clarification questions. "
                "Reference specific details from the user's feedback and avoid generic or previously asked questions. "
                "Each question should address a different aspect of the user's situation (for example: emotions, practical details, or future goals). "
                "Avoid yes/no questions, be supportive, and do not repeat yourself or previous questions in this conversation.\n"
                f"{context}"
                f"User feedback: \"{feedback}\"\n\n"
                "Questions:\n"
                "1."
            ),
            (
                "Imagine you are supporting someone who just shared the following feedback. "
                "Write 3 unique, open-ended questions to help them clarify their thoughts and feelings. "
                "Be gentle, supportive, and specific. Reference details from their feedback and avoid generic questions.\n"
                f"{context}"
                f"{example_block}"
                f"User feedback: \"{feedback}\"\n\n"
                "Questions:\n"
                "1."
            )
        ]
        prompt = random.choice(prompt_templates)
        prompt = f"Please respond in {language}.\n" + prompt

        if self.model and self.tokenizer:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("LLM raw output:", response)
            with open("llm_outputs.log", "a") as f:
                f.write(f"PROMPT:\n{prompt}\nOUTPUT:\n{response}\n{'='*40}\n")

            # Extract questions (split by numbers)
            questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip()[0].isdigit()]

            # Fuzzy filter for uniqueness
            unique_questions = []
            for q in questions:
                if all(fuzz.ratio(q, uq) < 80 for uq in unique_questions):
                    unique_questions.append(q)
                if len(unique_questions) == 3:
                    break

            # Ensure proper formatting
            def format_question(q):
                q = q.strip()
                if not q.endswith('?'):
                    q += '?'
                return q[0].upper() + q[1:] if q else q

            formatted_questions = [format_question(q) for q in unique_questions]

            def get_previous_bot_questions(conversation):
                return [
                    msg['text'] for msg in conversation
                    if msg['sender'].lower() == 'bot'
                ]

            previous_questions = get_previous_bot_questions(conversation or [])
            filtered_questions = []
            for q in formatted_questions:
                if all(fuzz.ratio(q, prev_q) < 80 for prev_q in previous_questions):
                    filtered_questions.append(q)
            # Filter out questions similar to poorly rated ones
            final_questions = []
            for q in filtered_questions:
                if all(fuzz.ratio(q, bad_q) < 80 for bad_q in BAD_QUESTIONS):
                    final_questions.append(q)
            return final_questions, idx, mode
        else:
            questions = [
                f"What do you mean by '{feedback}'?",
                f"Can you clarify your preference regarding '{feedback}'?",
                f"Could you provide more details about '{feedback}'?"
            ]
            return questions, idx, mode

    def refine_feedback(self, feedback):
        return self.generate_clarification_questions(feedback)
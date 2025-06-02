from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rapidfuzz import fuzz

class QuestionGenerator:
    def __init__(self, llm_model=None):
        if llm_model is not None:
            self.model = llm_model["model"]
            self.tokenizer = llm_model["tokenizer"]
            self.device = llm_model["device"]
        else:
            self.model = None
            self.tokenizer = None
            self.device = "cpu"

    def generate_clarification_questions(self, feedback, conversation=None):
        """
        Generate clarification questions based on ambiguous human feedback and recent conversation context.

        Parameters:
        feedback (str): The ambiguous feedback from the user.
        conversation (list): List of dicts with 'sender' and 'text' keys (optional).

        Returns:
        list: A list of clarification questions to refine user preferences.
        """
        context = ""
        if conversation:
            # Use the last 3 exchanges for context
            context = "Recent conversation:\n" + "\n".join(
                f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation[-3:]
            ) + "\n"
        if self.model and self.tokenizer:
            prompt = (
                "You are a helpful, empathetic assistant. "
                "Given the following conversation and user feedback, generate 3 open-ended, non-repetitive, and actionable clarification questions. "
                "Each question should address a different aspect of the user's situation (for example: emotions, practical details, or future goals). "
                "Avoid yes/no questions, be supportive, and do not repeat yourself.\n"
                f"{context}"
                f"User feedback: \"{feedback}\"\n\n"
                "Questions:\n"
                "1."
            )
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

            unique_questions = [format_question(q) for q in unique_questions]

            return unique_questions
        else:
            # Fallback placeholder logic
            questions = [
                f"What do you mean by '{feedback}'?",
                f"Can you clarify your preference regarding '{feedback}'?",
                f"Could you provide more details about '{feedback}'?"
            ]
            return questions

    def refine_feedback(self, feedback):
        """
        Refine the feedback by generating questions and returning them.

        Parameters:
        feedback (str): The initial ambiguous feedback.

        Returns:
        list: A list of refined questions based on the feedback.
        """
        return self.generate_clarification_questions(feedback)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

    def generate_clarification_questions(self, ambiguous_feedback):
        """
        Generate clarification questions based on ambiguous human feedback.

        Parameters:
        ambiguous_feedback (str): The ambiguous feedback from the user.

        Returns:
        list: A list of clarification questions to refine user preferences.
        """
        if self.model and self.tokenizer:
            prompt = (
                f"The user gave the following ambiguous feedback: '{ambiguous_feedback}'. "
                "Generate 3 specific clarification questions to help understand their true preference, each on a new line."
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated questions (after the prompt)
            generated = response[len(prompt):].strip()
            questions = [q.strip("- ").strip() for q in generated.split('\n') if q.strip()]
            return questions[:3] if questions else [generated]
        else:
            # Fallback placeholder logic
            questions = [
                f"What do you mean by '{ambiguous_feedback}'?",
                f"Can you clarify your preference regarding '{ambiguous_feedback}'?",
                f"Could you provide more details about '{ambiguous_feedback}'?"
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
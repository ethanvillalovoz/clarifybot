import torch

class RewardModel:
    def __init__(self, num_templates=2):
        # Bayesian reward model for each template
        self.alpha = [1 for _ in range(num_templates)]  # Prior successes (thumbs up)
        self.beta = [1 for _ in range(num_templates)]   # Prior failures (thumbs down)

    def update(self, template_idx, user_reward):
        """Update the Beta posterior for the template given binary feedback."""
        if user_reward == 1:
            self.alpha[template_idx] += 1
        else:
            self.beta[template_idx] += 1

    def posterior_mean(self, idx):
        return self.alpha[idx] / (self.alpha[idx] + self.beta[idx])

    def posterior_std(self, idx):
        a, b = self.alpha[idx], self.beta[idx]
        denom = (a + b) ** 2 * (a + b + 1)
        return (a * b / denom) ** 0.5 if denom > 0 else 0

    def summarize_preferences(self, conversation, llm_model=None):
        """
        Summarize the user's main concerns, feelings, and goals based on the conversation.
        """
        full_conversation = "\n".join(
            f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation
        )
        prompt = (
            "Summarize the user's main concerns, feelings, and goals based on the following conversation. "
            "Be concise and only include points that the user actually mentioned.\n"
            f"{full_conversation}\n"
            "Summary:"
        )
        tokenizer = llm_model["tokenizer"]
        model = llm_model["model"]
        device = llm_model["device"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the summary part
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()
        return summary

    def infer_reward(self, feedback):
        """
        Infer a reward value from a list of feedback (e.g., [1, 0, 1]).
        Returns the mean of the feedback as the reward.
        Raises ValueError if feedback is empty.
        """
        if not feedback:
            raise ValueError("Feedback cannot be empty.")
        return round(sum(feedback) / len(feedback), 2)

    # Placeholder for future reward function updates
    # def update_reward_function(self, new_feedback):
    #     pass

    # Placeholder for evaluating the reward function for a given state
    # def evaluate(self, state):
    #     return 0  # Default reward value

# Additional utility functions for reward modeling can be added here

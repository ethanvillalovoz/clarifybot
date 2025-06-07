import torch

class RewardModel:
    def __init__(self, num_templates=2):
        self.sums = [0.0 for _ in range(num_templates)]
        self.counts = [0 for _ in range(num_templates)]
        self.sq_sums = [0.0 for _ in range(num_templates)]  # For std

    def update(self, idx, reward):
        self.sums[idx] += reward
        self.sq_sums[idx] += reward ** 2
        self.counts[idx] += 1

    def posterior_mean(self, idx):
        return self.sums[idx] / self.counts[idx] if self.counts[idx] > 0 else 0.0

    def posterior_std(self, idx):
        if self.counts[idx] > 1:
            mean = self.posterior_mean(idx)
            variance = (self.sq_sums[idx] / self.counts[idx]) - mean ** 2
            return max(variance, 0) ** 0.5
        else:
            return 0.0

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

import unittest
from src.rl.reward_model import RewardModel

class TestRewardModel(unittest.TestCase):

    def setUp(self):
        self.reward_model = RewardModel()

    def test_infer_reward_from_positive_feedback(self):
        feedback = [1, 1, 1]  # Example of positive feedback
        expected_reward = 1.0  # Expected reward value
        inferred_reward = self.reward_model.infer_reward(feedback)
        self.assertAlmostEqual(inferred_reward, expected_reward, places=2)

    def test_infer_reward_from_negative_feedback(self):
        feedback = [0, 0, 0]  # Example of negative feedback
        expected_reward = 0.0  # Expected reward value
        inferred_reward = self.reward_model.infer_reward(feedback)
        self.assertAlmostEqual(inferred_reward, expected_reward, places=2)

    def test_infer_reward_with_mixed_feedback(self):
        feedback = [1, 0, 1]  # Example of mixed feedback
        expected_reward = 0.67  # Expected reward value based on the feedback
        inferred_reward = self.reward_model.infer_reward(feedback)
        self.assertAlmostEqual(inferred_reward, expected_reward, places=2)

    def test_infer_reward_with_empty_feedback(self):
        feedback = []  # No feedback
        with self.assertRaises(ValueError):
            self.reward_model.infer_reward(feedback)

if __name__ == '__main__':
    unittest.main()
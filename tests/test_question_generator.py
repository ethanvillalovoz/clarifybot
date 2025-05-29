import unittest
from src.llm.question_generator import QuestionGenerator

class TestQuestionGenerator(unittest.TestCase):

    def setUp(self):
        self.question_generator = QuestionGenerator()

    def test_generate_clarification_question(self):
        ambiguous_feedback = "I like the idea, but I'm not sure about the details."
        expected_question = "What specific details are you unsure about?"
        generated_question = self.question_generator.generate_clarification_question(ambiguous_feedback)
        self.assertEqual(generated_question, expected_question)

    def test_generate_clarification_question_empty_input(self):
        ambiguous_feedback = ""
        expected_question = "Could you please provide more details about your feedback?"
        generated_question = self.question_generator.generate_clarification_question(ambiguous_feedback)
        self.assertEqual(generated_question, expected_question)

    def test_generate_clarification_question_complex_input(self):
        ambiguous_feedback = "The concept is interesting, but the execution seems off."
        expected_question = "What aspects of the execution do you find off?"
        generated_question = self.question_generator.generate_clarification_question(ambiguous_feedback)
        self.assertEqual(generated_question, expected_question)

if __name__ == '__main__':
    unittest.main()
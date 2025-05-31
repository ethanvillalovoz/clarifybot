# main.py

from rl.reward_model import RewardModel
from llm.question_generator import QuestionGenerator
from gui.app import run_gui

def main():
    reward_model = RewardModel()
    question_generator = QuestionGenerator(llm_model=None)
    run_gui(reward_model, question_generator)

if __name__ == "__main__":
    main()
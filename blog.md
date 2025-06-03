# ClarifyBot: An LLM-Guided Clarification System for Ambiguous Reward Preferences

## What I Built
ClarifyBot is an innovative system designed to enhance the interaction between humans and autonomous agents by addressing the challenges of ambiguous human feedback. The core functionality of ClarifyBot revolves around its ability to receive noisy or unclear feedback from users and generate targeted clarification questions using a large language model (LLM). This process aims to refine the understanding of user preferences, ultimately leading to more aligned and effective autonomous behavior.

The project consists of several key components:
- **Reward Modeling Pipeline**: This component infers reward functions from human feedback, utilizing techniques such as inverse reinforcement learning to better understand user objectives.
- **Question Generation**: Leveraging an LLM, the system generates relevant clarification questions based on the ambiguous feedback received, facilitating a more interactive and informative dialogue with users.
- **Graphical User Interface**: The user-friendly GUI allows users to easily interact with the system, providing feedback and receiving clarification questions in real-time.

## Recent Improvements

- **Implemented `infer_reward` Method**: Added the `infer_reward` method to the [`RewardModel`](src/rl/reward_model.py) to properly infer reward values from user feedback, ensuring all reward model tests pass.
- **Requirements Update**: Updated `requirements.txt` to include all necessary dependencies, such as `rapidfuzz`, to ensure smooth installation and execution.
- **Testing and Validation**: Ran and passed all unit tests for both the reward model and question generator, confirming the correctness of the core logic.
- **Codebase Cleanup**: Reverted unnecessary changes, removed unused code, and improved consistency across modules.
- **Documentation**: Updated documentation and code comments to reflect the latest changes and clarify usage for future contributors.

## Why It Matters
As autonomous systems become more integrated into our daily lives, ensuring that these systems understand and align with human expectations is crucial. Traditional methods of learning from human feedback often treat users as perfect oracles, which is far from reality. ClarifyBot addresses this gap by acknowledging the imperfections and nuances of human feedback. By generating clarification questions, the system not only improves the quality of the data it learns from but also fosters a more collaborative relationship between humans and machines.

This project contributes to the broader field of human-robot interaction by exploring how autonomous agents can better understand and adapt to the complex and evolving preferences of their users. It highlights the importance of interactive feedback modeling and sets the stage for future research in this area.

## What I Learned
Throughout the development of ClarifyBot, I gained valuable insights into several key areas:
- **Human Feedback Complexity**: I learned that human feedback is often noisy and context-dependent, which necessitates a more sophisticated approach to understanding user preferences.
- **LLM Capabilities**: Working with large language models revealed their potential for generating meaningful and contextually relevant questions, enhancing the interaction between users and autonomous systems.
- **Reward Inference Techniques**: I deepened my understanding of reward modeling and inverse reinforcement learning, which are essential for aligning autonomous behavior with human objectives.
- **Importance of Testing and Dependency Management**: Ensuring all dependencies are tracked and all tests pass is crucial for maintainability and collaboration.

Overall, this project has not only strengthened my technical skills in machine learning and human-in-the-loop modeling but has also reinforced my commitment to developing autonomous systems that prioritize human alignment and safety.
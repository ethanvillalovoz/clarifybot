# ClarifyBot: An LLM-Guided Clarification System for Ambiguous Reward Preferences

## What I Built

ClarifyBot is an innovative system designed to enhance the interaction between humans and autonomous agents by addressing the challenges of ambiguous human feedback. The core functionality of ClarifyBot revolves around its ability to receive noisy or unclear feedback from users and generate targeted clarification questions using a large language model (LLM). This process aims to refine the understanding of user preferences, ultimately leading to more aligned and effective autonomous behavior.

The project consists of several key components:
- **Reward Modeling Pipeline**: Infers reward functions from human feedback, utilizing techniques such as inverse reinforcement learning to better understand user objectives.
- **Question Generation**: Leverages an LLM to generate relevant clarification questions based on ambiguous feedback, facilitating a more interactive and informative dialogue with users.
- **Graphical User Interface**: A user-friendly GUI allows users to easily interact with the system, provide feedback, and receive clarification questions in real-time.
- **Live Analytics Notebook**: A Jupyter notebook enables real-time analysis and visualization of both simulated and real user feedback, supporting research and debugging.

## How the RL Loop Works

ClarifyBot uses an explicit reinforcement learning (RL) loop—specifically, an epsilon-greedy bandit algorithm—to adaptively select which question template or prompt style to use for each clarification. At each step, the system either explores a new template (with probability epsilon) or exploits the template with the highest estimated reward. User feedback (e.g., thumbs up/down) is logged and used to update the reward estimates for each template. This approach enables ClarifyBot to learn, over time, which question styles are most effective for different users and situations, even when feedback is imperfect or noisy.

## Visualizing RL Adaptation

The analytics notebook includes plots that show:
- **Template Selection Frequencies Over Time**: How often each template is chosen, revealing adaptation and learning.
- **Average Rewards for Each Template**: How the system’s reward estimates for each template evolve as more feedback is collected.
- **Exploration vs. Exploitation**: How frequently the system explores new templates versus exploiting the best-known template.

## Recent Improvements

- **Robust Repeated Info Detection**: Refined the repeated information detection logic to prevent false positives, ensuring users are not incorrectly told they are repeating themselves. This greatly improved the user experience and allowed the RL loop to function as intended.
- **Consistent Logging for Analytics**: Standardized the logging format for all user interactions, guaranteeing that every event (template selection, mode, reward, timestamp) is recorded in a way that supports seamless analytics in the Jupyter notebook.
- **UI and Notebook Integration**: Achieved smooth integration between the web UI and the analytics notebook, so that real user interactions are immediately available for analysis and visualization.
- **Live Analytics and Visualization**: Fixed issues with blank or broken plots in the notebook by ensuring the log file is always in the expected format and by adding robust error handling in the analytics code.
- **Testing and Debugging**: Improved the debugging process by adding diagnostics to both the backend and the notebook, making it easier to trace issues with logging or repeated info detection.
- **User Experience Fixes**: Addressed issues where users were incorrectly told they were repeating themselves, resulting in a smoother and more engaging interaction.
- **Bandit Algorithm Ablation**: Implemented and compared both epsilon-greedy and UCB bandit algorithms in the analytics notebook, allowing for research-grade RLHF experiments.

## Why It Matters

As autonomous systems become more integrated into our daily lives, ensuring that these systems understand and align with human expectations is crucial. Traditional methods of learning from human feedback often treat users as perfect oracles, which is far from reality. ClarifyBot addresses this gap by acknowledging the imperfections and nuances of human feedback. By generating clarification questions, the system not only improves the quality of the data it learns from but also fosters a more collaborative relationship between humans and machines.

This project contributes to the broader field of human-robot interaction by exploring how autonomous agents can better understand and adapt to the complex and evolving preferences of their users. It highlights the importance of interactive feedback modeling and sets the stage for future research in this area.

## What I Learned

Throughout the development of ClarifyBot, I gained valuable insights into several key areas:
- **Human Feedback Complexity**: I learned that human feedback is often noisy and context-dependent, which necessitates a more sophisticated approach to understanding user preferences.
- **LLM Capabilities**: Working with large language models revealed their potential for generating meaningful and contextually relevant questions, enhancing the interaction between users and autonomous systems.
- **Reward Inference Techniques**: I deepened my understanding of reward modeling and inverse reinforcement learning, which are essential for aligning autonomous behavior with human objectives.
- **Importance of Testing and Dependency Management**: Ensuring all dependencies are tracked and all tests pass is crucial for maintainability and collaboration.
- **Iterative Debugging and Integration**: Real-world systems require iterative debugging and refinement, especially when integrating multiple components (UI, backend, analytics). Consistent, well-structured logging is crucial for reliable analytics and visualization.

Overall, this project has not only strengthened my technical skills in machine learning and human-in-the-loop modeling but has also reinforced my commitment to developing autonomous systems that prioritize human alignment and safety.

## Limitations and Future Work

- **Template Diversity**: Currently, the system uses a fixed set of question templates. Future work could explore dynamic template generation or fine-tuning LLMs for clarification.
- **Bandit Algorithm Scope**: While both epsilon-greedy and UCB are implemented, more advanced algorithms (e.g., Thompson Sampling) could be explored.
- **User Modeling**: The system does not yet personalize to individual users or contexts; adding user modeling could further improve alignment.
- **Scalability**: The current implementation is designed for research and demonstration; scaling to production would require further engineering and evaluation.
- **Long-Term Adaptation**: Investigating how the system adapts over longer time horizons and with more diverse feedback would provide deeper insights.
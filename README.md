# ClarifyBot: An LLM-Guided Clarification System for Ambiguous Reward Preferences

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ethanvillalovoz/clarifybot/blob/main/notebooks/demo.ipynb)

## Overview

ClarifyBot is an interactive research system that demonstrates how autonomous agents can learn from imperfect human feedback by refining reward inference through clarification questions. Leveraging large language models (LLMs) and a simple reinforcement learning (RL) loop, ClarifyBot adaptively selects question-generation strategies based on user feedback, making the agent more aligned with human expectations.

## Features

- **Interactive Feedback Modeling:** Generates clarification questions in response to ambiguous or noisy human feedback, enabling more accurate understanding of user preferences.
- **RL-Driven Adaptive Question Generation:** Uses a bandit-style RL loop to adaptively select among multiple question templates or prompt styles, learning which approaches yield the most useful clarifications based on user ratings.
- **Reward Inference and Analytics:** Aggregates user feedback (e.g., thumbs up/down) into a reward signal, which is tracked and visualized for each question template.
- **Misalignment Quantification:** Quantifies its confidence in user alignment and warns when it is uncertain, surfacing misalignment to the user.
- **Live Visualization:** The GUI includes a real-time bar chart showing both the average reward and usage count for each question template, making the learning process transparent.
- **User-Friendly Interface:** The graphical user interface (GUI) allows seamless interaction, feedback, and visualization of the system’s learning process.
- **Demonstration Notebook:** A Jupyter notebook is included to showcase ClarifyBot’s capabilities and analytics.

## Recent Improvements

- **Improved Repeated Info Detection:** The system now uses a more robust and configurable method for detecting repeated user messages, reducing false positives and improving the flow of conversation.
- **Reliable Logging for Analytics:** All user interactions are logged with complete information (template index, mode, reward, timestamp), ensuring that analytics and visualizations in the notebook are always accurate and up-to-date.
- **UI ↔ Notebook Analytics Integration:** Real user feedback from the UI is now immediately available for analysis in the Jupyter notebook, enabling live research and debugging.
- **Enhanced Error Handling:** The analytics notebook now gracefully handles missing or malformed log data, providing clear diagnostics instead of blank plots or errors.
- **User Experience Fixes:** Addressed issues where users were incorrectly told they were repeating themselves, resulting in a smoother and more engaging interaction.

## Research Context

ClarifyBot is inspired by research in interactive reward modeling and alignment (see [Andreea Bobu’s work](https://www.mit.edu/~abobu/)), addressing the challenges of learning from noisy, incomplete, or inconsistent human feedback. The system demonstrates:

- How clarification can improve reward inference compared to passive feedback collection.
- How simple RL techniques can adapt agent behavior to maximize alignment with human preferences.
- How to quantify and visualize misalignment/confidence in real time.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/clarifybot.git
   cd clarifybot
   ```

2. **(Recommended) Create and activate a Conda environment:**
   ```bash
   conda create -n clarifybot python=3.9
   conda activate clarifybot
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*If you do not use Conda, you can install dependencies directly into your Python environment as usual.*

## Usage

To run the ClarifyBot application, execute:
```bash
python src/main.py
```
Once running, open the GUI in your browser to interact with the system, provide feedback, and see live analytics.

## Example

Refer to the [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a detailed demonstration of how ClarifyBot processes feedback, generates clarification questions, and adapts its strategy based on user ratings. The notebook also includes analytics and visualizations for both simulated and real user data.

## Testing

Unit tests are provided for reward modeling and question generation. To run the tests:
```bash
pytest tests/
```

## Contribution

Contributions are welcome! Please submit a pull request or open an issue for suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
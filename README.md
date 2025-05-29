# ClarifyBot: An LLM-Guided Clarification System for Ambiguous Reward Preferences

## Overview
ClarifyBot is an innovative system designed to enhance the interaction between humans and autonomous agents by refining reward preferences through clarification questions. By leveraging large language models (LLMs), ClarifyBot addresses the challenges of learning from imperfect human feedback, ensuring that autonomous systems align more closely with human expectations.

## Features
- **Interactive Feedback Modeling**: ClarifyBot generates clarification questions based on ambiguous or noisy human feedback, allowing for a more accurate understanding of user preferences.
- **Reward Inference**: The system employs a reward modeling pipeline to infer reward functions from human feedback, utilizing techniques such as inverse reinforcement learning.
- **User-Friendly Interface**: The graphical user interface (GUI) facilitates seamless interaction, displaying feedback and generated questions in an intuitive manner.
- **Demonstration Notebook**: A Jupyter notebook is included to showcase the functionality of ClarifyBot, allowing users to see the system in action.

## Installation
To set up ClarifyBot, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/clarifybot.git
   cd clarifybot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the ClarifyBot application, execute the following command:
```
python src/main.py
```

Once the application is running, you can interact with the system through the GUI, providing feedback and receiving clarification questions.

## Example
Refer to the `notebooks/demo.ipynb` for a detailed demonstration of how ClarifyBot processes feedback and generates clarification questions.

## Testing
Unit tests are provided to ensure the functionality of the reward modeling and question generation components. To run the tests, use:
```
pytest tests/
```

## Contribution
Contributions to the ClarifyBot project are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
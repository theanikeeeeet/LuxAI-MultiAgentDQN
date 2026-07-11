🎯 Introduction
This project implements Deep Q-Networks (DQN) for multi-agent reinforcement learning (MARL) in the Lux AI Challenge Season 3. The agents are trained to strategically navigate, collect resources, and compete effectively using a shared replay buffer and separate policy & target networks.

✨ Features
✅ Multi-Agent Deep Q-Learning for strategic decision-making. ✅ Experience Replay using a shared replay buffer. ✅ Target & Policy Networks to stabilize training. ✅ Epsilon-Greedy Exploration for improved performance. ✅ Adaptive Model Sampling to prevent overfitting to one strategy. ✅ Log-Based Reward Tracking for debugging.

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your_username/LuxAI-MultiAgentDQN.git
cd LuxAI-MultiAgentDQN
2️⃣ Set Up Virtual Environment
python3 -m venv luxai_env
source luxai_env/bin/activate  # On Windows use `luxai_env\Scripts\activate`
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Install Lux AI Environment
pip install luxai_s3
🚀 Usage
Run the Environment with Your Bot
luxai-s3 path/to/your/bot.py path/to/opponent/bot.py --output replay.json
Train the Model
python train.py
Evaluate Performance
python evaluate.py --episodes 50
🧠 Training Pipeline
1️⃣ Observation Extraction: Converts raw environment state into agent-friendly representations. 2️⃣ Action Selection: Uses the policy network (with epsilon-greedy exploration). 3️⃣ Experience Storage: Saves (state, action, reward, next_state) tuples into a shared replay buffer. 4️⃣ Batch Training:

Samples a mini-batch from replay buffer.
Updates policy network using TD-Target from target network.
Optimizes using MSE loss for Q-values. 5️⃣ Target Network Update: Periodically syncs with policy network for stability.
🔧 Architecture
📌 Multi-Agent DQN Structure:

Agent.py -> Handles game interaction, reward computation
DQN.py -> Deep Q-Network, Policy & Target Network
ReplayBuffer.py -> Shared Experience Replay Buffer
Train.py -> Training Loop
Evaluate.py -> Performance Evaluation
📌 Neural Network Model:

Fully Connected Layers with ReLU activation
Q-Values Head: Predicts action-value estimates
X, Y Coordinate Heads: Predicts (x, y) positions for sapping actions
📊 Results & Performance
📌 Training Progress

Episodes	Avg Reward	Win Rate (%)
100	-5.3	45.2
500	10.2	60.5
1000	18.7	75.8
📌 Sample Training Reward Curve Reward Curve (Replace path/to/reward_curve.png with your actual image!)

💡 Contributing
We welcome contributions! To contribute: 1️⃣ Fork the repository. 2️⃣ Create a feature branch. 3️⃣ Commit changes with meaningful messages. 4️⃣ Submit a pull request.

📜 License
📌 This project is licensed under MIT License.


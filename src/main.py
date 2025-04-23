import torch
from RL.buffer.replay_buffer import ReplayBuffer
from RL.agents.dqn_agent import DQNAgent
from RL.solitare_env import SolitaireEnv  # Your environment wrapper

# Hyperparameters
INPUT_SIZE = 29
OUTPUT_SIZE = 83
HIDDEN_SIZE = 128
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 64
EPISODES = 1000
MAX_STEPS = 500

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))

env = SolitaireEnv()
buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
agent = DQNAgent(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, buffer=buffer, device=device)

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.select_action(state_tensor)

        next_state, reward, done, _ = env.step(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        buffer.push(state_tensor, action, reward, next_state_tensor, float(done))

        loss = agent.train_step(BATCH_SIZE)
        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Loss: {loss:.4f}" if loss else f"Episode {episode + 1} | Total Reward: {total_reward:.2f}")

# Save the trained model
torch.save(agent.q_network.state_dict(), "dqn_solitaire.pth")
print("Model saved as dqn_solitaire.pth")
import torch
from RL.buffer.replay_buffer import ReplayBuffer
from RL.agents.dqn_agent import DQNAgent
from RL.solitare_env import SolitaireEnv  # Your environment wrapper
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Hyperparameters
INPUT_SIZE = 204
OUTPUT_SIZE = 616
HIDDEN_SIZE = 128
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 64
EPISODES = 10
MAX_STEPS = 500
MODEL_PATH = "TrainedModels/checkpoint.pth"

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))

env = SolitaireEnv()
buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
agent = DQNAgent(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, buffer=buffer, device=device)

# Load checkpoint if it exists
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.target_network.load_state_dict(checkpoint["target_network"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.epsilon = checkpoint["epsilon"]
    print(f"✅ Loaded checkpoint from {MODEL_PATH}")

# Track statistics
episode_rewards = []
episode_losses = []
epsilon_values = []
num_wins = 0

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    SolitaireEnv.decode_observation(state)
    total_reward = 0
    win = False
    loss = None

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

    episode_rewards.append(total_reward)
    episode_losses.append(loss if loss is not None else 0)
    epsilon_values.append(agent.epsilon)
    if win:
        num_wins += 1

    win_rate = num_wins / (episode + 1)

    env.render()
    print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Loss: {loss:.4f}" if loss else f"Episode {episode + 1} | Total Reward: {total_reward:.2f}")

# Save checkpoint
torch.save({
    "q_network": agent.q_network.state_dict(),
    "target_network": agent.target_network.state_dict(),
    "optimizer": agent.optimizer.state_dict(),
    "epsilon": agent.epsilon,
}, MODEL_PATH)
print(f"✅ Checkpoint saved to {MODEL_PATH}")

# Save plots with timestamp
os.makedirs("Plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"Plots/training_plot_{timestamp}.png"

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(episode_rewards, label="Reward")
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(episode_losses, label="Loss")
plt.title("Episode Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(plot_path)
plt.show()
print(f"Plot saved as {plot_path}")
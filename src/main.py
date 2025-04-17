from RL.solitare_env import SolitaireEnv
from RL.simple_agent import SimpleAgent

def main():
    env = SolitaireEnv()
    agent = SimpleAgent(env.actions)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # optional: show board

    print(f"Game over. Total reward: {total_reward}")

if __name__ == "__main__":
    main()

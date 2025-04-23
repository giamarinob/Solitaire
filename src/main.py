from time import sleep
from RL.solitare_env import SolitaireEnv
from RL.agents.simple_agent import SimpleAgent

def main():
    env = SolitaireEnv()
    agent = SimpleAgent(env.action_space)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        # print(env.action_space[action])
        obs, reward, done, info = env.step(action)
        SolitaireEnv.decode_observation(obs)
        total_reward += reward
        # env.render()  # optional: show board
        sleep(1)

    print(f"Game over. Total reward: {total_reward}")

if __name__ == "__main__":
    main()

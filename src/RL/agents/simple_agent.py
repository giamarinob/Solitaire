import random

class SimpleAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        # For now, select random action (you can later plug in logic)
        return random.randint(0, len(self.action_space) - 1)

    def learn(self, *args, **kwargs):
        # Placeholder for training logic (Q-learning, DQN, etc.)
        pass

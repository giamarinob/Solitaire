# src/rl/solitaire_env.py
from src.Solitaire import Solitaire
import copy

class SolitaireEnv:
    def __init__(self):
        self.game = Solitaire()
        self.action_space = self._build_action_space()

    def _build_action_space(self):
        actions = []

        actions.append(("draw",))
        actions.append(("undo",))
        actions.append(("quit",))
        for i in range(7):
            actions.append(("waste_to_tableau", i))
        for s in ["hearts", "diamonds", "clubs", "spades"]:
            actions.append(("waste_to_foundation", s))
        for i in range(7):
            actions.append(("tableau_to_foundation", i))
        for from_idx in range(7):
            for to_idx in range(7):
                if from_idx != to_idx:
                    actions.append(("tableau_to_tableau", (from_idx, to_idx)))
        for suit in ["hearts", "diamonds", "clubs", "spades"]:
            for to_idx in range(7):
                actions.append(("foundation_to_tableau", (suit, to_idx)))

        return actions

    def reset(self):
        self.game = Solitaire()
        return self._get_observation()

    def step(self, action):
        # Execute the given action tuple
        reward = 0
        done = False
        info = {}

        move_successful = self._execute_action(action)

        if not move_successful:
            reward = -5  # Penalty for illegal move
        else:
            reward = self._calculate_reward(action)

        if self.game.is_won():
            done = True
            self.game.end()  # Calculate bonus score
            reward += 1000  # Big bonus

        return self._get_observation(), reward, done, info

    def _execute_action(self, action):
        # Deconstruct and map to your game's internal methods
        # e.g., if action[0] == "draw_from_stock" -> call self.game.draw()
        pass

    def _calculate_reward(self, action):
        # Could use score delta, or fixed values like your current design
        return 0

    def _get_observation(self):
        # Build a representation of the game state
        # You can return a custom object or encode it for NN input
        return copy.deepcopy(self.game)

    def render(self):
        self.game.display()  # Or whatever your display function is

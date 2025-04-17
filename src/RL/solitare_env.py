# src/rl/solitaire_env.py
from src.Solitaire import Solitaire
import copy

class SolitaireEnv:
    def __init__(self):
        self.game = Solitaire()
        self.action_space = self._build_action_space()

    def _build_action_space(self):
        actions = []

        # Action: Draw from stock
        actions.append(("draw",))

        # Waste to Foundation/Tableau
        for i in range(4):
            actions.append(("w2f", i))
        for i in range(7):
            actions.append(("w2t", i))

        # Tableau to Foundation
        for tableau_idx in range(7):
            for foundation_idx in range(4):
                actions.append(("t2f", tableau_idx, foundation_idx))

        # Tableau to Tableau (top card only)
        for i in range(7):
            for j in range(7):
                if i != j:
                    actions.append(("t2t", i, j))

        # Optional: Foundation to Tableau
        # for i in range(4):
        #     for j in range(7):
        #         actions.append(("f2t", i, j))

        # Optional: Grant Access to Undo Function
        # actions.append(("undo")

        # Optional: Quit
        actions.append(("quit",))

        return actions

    def reset(self):
        self.game = Solitaire()
        return self._get_observation()

    def step(self, action):
        # Execute the given action tuple
        reward = 0
        done = False
        info = {}

        move_successful = self._execute_action(
            self.action_space[action]
        )

        if not move_successful:
            reward = -5  # Penalty for illegal move
        else:
            reward = self._calculate_reward(action)

        if self.game.is_won():
            done = True
            reward += 1000  # Big bonus
        elif self.action_space[action][0] == 'quit':
            done = True

        return self._get_observation(), reward, done, info

    def _execute_action(self, action):
        try:
            if action[0] == "draw":
                self.game.draw_from_stock()
            elif action[0] == "w2f":
                return self.game.move("waste", None, "foundation", action[1])
            elif action[0] == "w2t":
                return self.game.move("waste", None, "tableau",action[1])
            elif action[0] == "t2f":
                return self.game.move("tableau", action[1], "foundation", action[2])
            elif action[0] == "t2t":
                # Simplified version will only move the top card on a tableau
                return self.game.move("tableau", action[1], "tableau", action[2], 0)
            elif action[0] == "f2t":
                # Not currently exposed in the action space
                return self.game.move("foundation", action[1], "tableau", action[2])
            elif action[0] == "undo":
                # Not currently exposed in the action space
                return self.game.undo()
            elif action[0] == "quit":
                self.game.end()
                return True
        except Exception as e:
            print(f"Action {action} caused error: {e}")
            return False
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

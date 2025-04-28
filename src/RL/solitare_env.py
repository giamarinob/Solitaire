# src/rl/solitaire_env.py
from src.Solitaire import Solitaire
from src.Solitaire.card import Card
import copy

class SolitaireEnv:
    @staticmethod
    def decode_observation(obs):
        suit_values = list(Card.SUITS.values())

        def decode_card(value, suit_idx):
            if value == 0:
                return "[--]"
            suit = suit_values[suit_idx] if 0 <= suit_idx < len(suit_values) else "?"
            rank = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}.get(value, str(value))
            return f"[{rank}{suit}]"

        sections = {
            "waste": obs[:6],
            "foundations": obs[6:14],
            "tableaus": obs[14:28],
            "stock_size": obs[28],
        }

        print("Decoded Observation: ")

        print("Waste:")
        for i in range(0, 6, 2):
            print(" ", decode_card(sections["waste"][i], sections["waste"][i + 1]))

        print("Foundations:")
        for i in range(0, 8, 2):
            print(" ", decode_card(sections["foundations"][i], sections["foundations"][i + 1]))

        print("Tableaus:")
        for i in range(0, 14, 2):
            print(f"  Pile {i // 2 + 1}:", decode_card(sections["tableaus"][i], sections["tableaus"][i + 1]))

        print(f"Stock size: {sections['stock_size']}")

    @staticmethod
    def _suit_to_index(suit):
        try:
            suit_order = list(Card.SUITS.values())  # ensure consistent order
            return suit_order.index(suit)
        except (ImportError, ValueError):
            return 0  # fallback index if suit not found

    def __init__(self):
        self.game = Solitaire()
        self.action_space = self._build_action_space()
        self.prev_score = 0

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
            reward = self._calculate_reward()

        if self.game.is_won():
            done = True
            reward += 10000  # Big bonus
        elif self.action_space[action][0] == 'quit':
            reward -= 10 # Small penalty for quitting early
            done = True

        return self._get_observation(), reward, done, info

    def _execute_action(self, action):
        print("Executing action: ", action)
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
            self.game.display()
            return False
        pass

    def _calculate_reward(self):
        # Only use base_score to compute learning reward
        prev_score = self.prev_score
        current_score = self.game.base_score
        reward = current_score - prev_score
        self.prev_score = current_score  # update for next step
        return reward

    def _get_observation(self):
        obs = []

        # Waste (top 3 cards, from bottom to top)
        top_waste = self.game.waste.get_top_three()
        for i in range(3):
            if i < len(top_waste):
                card = top_waste[i]
                obs.extend([card.value, SolitaireEnv._suit_to_index(card.suit)])
            else:
                obs.extend([0, 0])

        # Foundations (top card from each foundation pile)
        for foundation in self.game.foundations:
            top_card = foundation.top_card()
            if top_card:
                obs.extend([
                    top_card.value,
                    SolitaireEnv._suit_to_index(top_card.suit)
                ])
            else:
                obs.extend([0, 0])

        # Tableaus (top card from each pile only)
        for tableau in self.game.tableaus:
            top_card = tableau.top_card()
            if top_card and top_card.face_up:
                obs.extend([
                    top_card.value,
                    SolitaireEnv._suit_to_index(top_card.suit)
                ])
            else:
                obs.extend([0, 0])

        # Stock size
        obs.append(len(self.game.stock.cards))

        return obs

    def render(self):
        self.game.display()  # Or whatever your display function is

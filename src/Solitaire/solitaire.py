import copy
import time
from src.Solitaire.deck import Deck
from src.Solitaire.stock import Stock
from src.Solitaire.waste import Waste
from src.Solitaire.tableau import Tableau
from src.Solitaire.foundation import Foundation

class Solitaire:
    def __init__(self):
        # Initialize game objects
        self.deck = Deck()
        self.stock = Stock()
        self.waste = Waste()
        self.tableaus = [Tableau() for _ in range(7)]
        self.foundations = [Foundation() for _ in range(4)]
        self.score = 0  # Starting score

        # Used for undoing a move
        self.undo_stock = None
        self.undo_waste = None
        self.undo_tableaus = None
        self.undo_foundations = None
        self.undo_score = None
        self.undo_flag = False

        # 5 Minute time limit
        self.start_time = time.time()
        self.time_limit = 300 # 5 minutes

        self.game_won = False

        self._setup_game()

    def _setup_game(self):
        self.deck.shuffle()

        # Deal cards to tableau piles
        for i, tableau in enumerate(self.tableaus):
            for j in range(i + 1):
                card = self.deck.deal()
                # Only the top card is face up
                if j == i:
                    card.flip()
                tableau.add_card(card)

        # Remaining cards go to stock
        while not self.deck.is_empty():
            self.stock.add_card(self.deck.deal())

    def time_remaining(self):
        elapsed = time.time() - self.start_time
        return max(0, int(self.time_limit - elapsed))

    def time_up(self):
        return self.time_remaining() <= 0

    def draw_from_stock(self):
        self._store_move(self.stock, self.waste, self.tableaus, self.foundations, self.score)

        if self.stock.is_empty():
            self._recycle_waste()
            self.score -= 20
            print("Recycled waste")
        else:
            # Draw up to 3 cards from stock into waste
            for _ in range(min(3, len(self.stock.cards))):
                card = self.stock.deal()
                self.waste.add_card(card)

    def _recycle_waste(self):
        # Move all waste cards back to stock when stock is empty
        if self.stock.is_empty():
            recycled = self.waste.recycle()
            for card in recycled:
                card.flip()
                self.stock.add_card(card)

    def can_move_tableau_to_tableau(self, from_index, to_index, start_card_index):
        # Check if it's legal to move a sequence of cards from one tableau to another.
        from_tableau = self.tableaus[from_index]
        to_tableau = self.tableaus[to_index]

        index = start_card_index - 1

        # Grab the card we're moving (must be face up and valid index)
        if index < 0 or index >= len(from_tableau.cards):
            return False

        moving_card = from_tableau.cards[index]
        if not moving_card.face_up:
            return False  # Cannot move facedown cards

        if to_tableau.is_empty():
            return moving_card.rank == 'K'  # Only Kings can be moved to empty tableaus

        target_card = to_tableau.top_card()
        return (
                moving_card.color != target_card.color and
                moving_card.value == target_card.value - 1
        )

    def move(self, source_type, source_index, dest_type, dest_index, start_index=None):
        """
        General move function used to dispatch all valid solitaire moves.

        source_type: 'tableau', 'waste', 'foundation'
        source_index: index in the list of source stacks
        dest_type: 'tableau', 'foundation'
        dest_index: index in the list of destination stacks
        start_index: for tableau-to-tableau moves, the index of the first card to move
        """

        if source_type == 'tableau' and dest_type == 'tableau':
            return self._move_tableau_to_tableau(source_index, dest_index, start_index)

        elif source_type == 'tableau' and dest_type == 'foundation':
            return self._move_tableau_to_foundation(source_index, dest_index)

        elif source_type == 'waste' and dest_type == 'tableau':
            return self._move_waste_to_tableau(dest_index)

        elif source_type == 'waste' and dest_type == 'foundation':
            return self._move_waste_to_foundation(dest_index)

        elif source_type == 'foundation' and dest_type == 'tableau':
            return self._move_foundation_to_tableau(source_index, dest_index)

        else:
            print(f"Invalid move request: {source_type} to {dest_type}")
            return False

    def _store_move(self, stock, waste, tableaus, foundations, score):
        self.undo_stock = copy.deepcopy(stock)
        self.undo_waste = copy.deepcopy(waste)
        self.undo_tableaus = copy.deepcopy(tableaus)
        self.undo_foundations = copy.deepcopy(foundations)
        self.undo_score = score
        self.undo_flag = True

    def undo(self):
        if self.undo_flag:
            self.stock = self.undo_stock
            self.waste = self.undo_waste
            self.tableaus = self.undo_tableaus
            self.foundations = self.undo_foundations
            self.score = self.undo_score
            self.undo_flag = False
            self.undo_stock = None
            self.undo_waste = None
            self.undo_foundations = None
            self.undo_score = None
            self.undo_tableaus = None
            return True

        return False

    def _move_tableau_to_tableau(self, from_index, to_index, start_index):
        if not self.can_move_tableau_to_tableau(from_index, to_index, start_index):
            print("Illegal move.")
            return False

        from_tableau = self.tableaus[from_index]
        to_tableau = self.tableaus[to_index]

        self._store_move(self.stock, self.waste, self.tableaus, self.foundations, self.score)
        # Slice out the cards to move
        index = start_index - 1

        cards_to_move = from_tableau.cards[index:]

        # Add to the destination tableau
        for card in cards_to_move:
            to_tableau.add_card(card)

        # Remove from the source tableau
        from_tableau.cards = from_tableau.cards[:index]

        # Flip new top card if needed
        if from_tableau.cards and not from_tableau.top_card().face_up:
            from_tableau.top_card().flip()
            self.score += 20

        return True

    def _move_tableau_to_foundation(self, tableau_index, foundation_index):
        tableau = self.tableaus[tableau_index]
        foundation = self.foundations[foundation_index]

        card = tableau.top_card()
        if not card or not card.face_up:
            print("No face-up card to move.")
            return False

        if not foundation.can_add_card(card):
            print("Illegal move to foundation.")
            return False

        self._store_move(self.stock, self.waste, self.tableaus, self.foundations, self.score)
        # Perform the move
        foundation.add_card(tableau.remove_card())
        self.score += 100

        # Flip new top card if needed
        if tableau.cards and not tableau.top_card().face_up:
            tableau.top_card().flip()
            self.score += 20

        return True

    def _move_waste_to_tableau(self, tableau_index):
        tableau = self.tableaus[tableau_index]
        card = self.waste.top_card()

        if not card:
            print("No card in waste to move.")
            return False

        if tableau.is_empty():
            if card.rank != 'K':
                print("Only Kings can be moved to an empty tableau.")
                return False
        else:
            target_card = tableau.top_card()
            if card.color == target_card.color or card.value != target_card.value - 1:
                print("Illegal move to tableau.")
                return False

        self._store_move(self.stock, self.waste, self.tableaus, self.foundations, self.score)
        # Perform the move
        tableau.add_card(self.waste.remove_top_card())
        self.score += 20
        return True

    def _move_waste_to_foundation(self, foundation_index):
        foundation = self.foundations[foundation_index]
        card = self.waste.top_card()

        if not card:
            print("No card in waste to move.")
            return False

        if not foundation.can_add_card(card):
            print("Illegal move to foundation.")
            return False

        self._store_move(self.stock, self.waste, self.tableaus, self.foundations, self.score)

        foundation.add_card(self.waste.remove_top_card())
        self.score += 120
        return True

    def _move_foundation_to_tableau(self, foundation_index, tableau_index):
        foundation = self.foundations[foundation_index]
        tableau = self.tableaus[tableau_index]

        card = foundation.top_card()
        if not card:
            print("No card in foundation to move.")
            return False

        if tableau.is_empty():
            if card.rank != 'K':
                print("Only Kings can be moved to an empty tableau.")
                return False
        else:
            top_tableau_card = tableau.top_card()
            if card.color == top_tableau_card.color or card.value != top_tableau_card.value - 1:
                print("Illegal move from foundation to tableau.")
                return False

        self._store_move(self.stock, self.waste, self.tableaus, self.foundations, self.score)

        tableau.add_card(foundation.remove_top_card())
        self.score -= 100
        return True

    def is_won(self):
        # Normal win
        if self._get_foundation_total() == 52:
            self.game_won = True
            self.end()
            return True

        # Deterministic win
        if self.is_deterministic_win():
            for tableau in self.tableaus:
                self.score += len(tableau.cards) * 100
            self.game_won = True
            self.end()
            return True

        return False

    def is_deterministic_win(self):
        if not self.stock.cards and not self.waste.cards:
            for tableau in self.tableaus:
                for card in tableau.cards:
                    if not card.face_up:
                        return False
            return True
        return False

    def end(self):
        time = self.time_remaining()

        if self.game_won:
            self.score += time * 100
        else:
            num_cards = self._get_foundation_total()

            if num_cards < 10:
                self.score += time
            elif num_cards < 20:
                self.score += time * 2
            elif num_cards < 35:
                self.score += time * 10
            else:
                self.score += time * 50

    def _get_foundation_total(self):
        total = 0
        for foundation in self.foundations:
            total += len(foundation.cards)

        return total

    def display(self):
        print("=== Solitaire Game State ===")
        remaining = self.time_remaining()
        minutes, seconds = divmod(remaining, 60)
        print(f"Time remaining: {minutes:02}:{seconds:02}")
        print(f"Score: {self.score}")
        print(f"Stock: {len(self.stock.cards)} cards")
        print(self.waste)
        print("\nTableaus:")
        for i, tableau in enumerate(self.tableaus):
            print(f"  {i}: {tableau}")
        print("\nFoundations:")
        for i, foundation in enumerate(self.foundations):
            print(f"  {i}: {foundation}")

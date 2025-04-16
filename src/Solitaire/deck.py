import random
from src.Solitaire.card import Card

class Deck:
    def __init__(self):
        # Create a new deck of 52 cards
        self.cards = [Card(suit, rank) for suit in Card.SUITS for rank in Card.RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        if len(self.cards) > 0:
            return self.cards.pop()
        else:
            raise ValueError("Deck is empty")

    def is_empty(self):
        return len(self.cards) == 0

    def __repr__(self):
        return f"Deck({len(self.cards)} cards remaining)"
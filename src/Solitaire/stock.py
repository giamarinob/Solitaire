class Stock:
    def __init__(self):
        """Initialize an empty stock."""
        self.cards = []  # List of cards in the stock

    def deal(self):
        """Deal a card from the stock."""
        if self.cards:
            card = self.cards.pop()
            return card
        return None

    def display(self):
        """Display the current stock for debugging purposes."""
        return [str(card) for card in self.cards]

    def add_card(self, card):
        """Add a card to the stock."""
        self.cards.append(card)

    def is_empty(self):
        return len(self.cards) == 0
class Tableau:
    def __init__(self):
        """
        Initialize a tableau with an empty list of cards.
        """
        self.cards = []  # Cards in this tableau pile

    def add_card(self, card):
        # Add a card to the tableau.
        self.cards.append(card)

    def remove_card(self):
        # Remove the top card from the tableau.
        if self.cards:
            return self.cards.pop()
        return None

    def is_empty(self):
        # Check if the tableau is empty.
        return len(self.cards) == 0

    def top_card(self):
        # Get the top card from the tableau (if any).
        if self.cards:
            return self.cards[-1]
        return None

    def display(self):
        # Display the current tableau for debugging purposes.
        return [str(card) for card in self.cards]

    def can_add_card(self, card):
        # Check if a card can be added to the tableau.
        # Placeholder for future move validation logic (e.g., alternating colors and descending order)
        pass

    def __repr__(self):
        return f"Tableau: {self.display()}"

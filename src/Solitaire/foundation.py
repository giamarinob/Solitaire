class Foundation:
    def __init__(self):
        """Initialize an empty foundation that can hold cards in any suit until an Ace is added."""
        self.cards = []  # Cards in this foundation pile
        self.suit = None  # The foundation suit (None means it's not locked yet)

    def add_card(self, card):
        """Add a card to the foundation pile if it's the correct next card in the sequence."""
        if self.can_add_card(card):
            self.cards.append(card)
            # After adding the Ace, lock the foundation to the card's suit
            if card.value == 1:
                self.suit = card.suit
            return True
        return False

    def can_add_card(self, card):
        """Check if the card can be added to the foundation."""
        top_card = self.top_card()
        if not top_card:
            # If foundation is empty, only Ace from any suit can be placed.
            return card.value == 1

        if self.suit and card.suit != self.suit:
            # If the foundation has a suit, the card must match the suit.
            print("Suit does not match " + self.suit)
            return False

        # The card must be one rank higher than the top card (ascending order)
        if card.value == top_card.value + 1:
            return True

        return False

    def remove_top_card(self):
        card = self.top_card()
        if card and card.value != 1:
            return self.cards.pop()
        return None

    def top_card(self):
        if self.cards:
            return self.cards[-1]
        return None

    def display(self):
        """Display the current foundation for debugging purposes."""
        return [str(card) for card in self.cards]

    def __repr__(self):
        suit_info = self.suit if self.suit is not None else "Empty"
        return f"Foundation({suit_info}): {self.display()}"

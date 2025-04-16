class Waste:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        if card:
            card.flip()
            self.cards.append(card)

    def top_card(self):
        if self.cards:
            return self.cards[-1]
        return None

    def get_top_three(self):
        """Returns a list of up to the top 3 cards from the waste, ordered from oldest to newest."""
        return self.cards[-3:] if len(self.cards) >= 3 else self.cards[:]

    def remove_top_card(self):
        if self.cards:
            return self.cards.pop()
        return None

    def recycle(self):
        recycled = self.cards[::-1]
        self.cards = []
        return recycled

    def display(self):
        return [str(card) for card in self.cards]

    def __repr__(self):
        top_three = self.get_top_three()
        if not top_three:
            return "Waste: [empty]"

        repr_lines = ["Waste (top to bottom):"]
        for i, card in enumerate(top_three):
            if i == len(top_three) - 1:
                repr_lines.append(f"  → {card}  ← top (playable)")
            else:
                repr_lines.append(f"    {card}")
        return "\n".join(repr_lines)
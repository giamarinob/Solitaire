class Card:
    SUITS = {
        'spades': '♠',
        'hearts': '♥',
        'diamonds': '♦',
        'clubs': '♣'
    }

    RANKS = ['A'] + [str(n) for n in range(2, 11)] + ['J', 'Q', 'K']

    COLORS = {
        'spades': 'black',
        'clubs': 'black',
        'hearts': 'red',
        'diamonds': 'red'
    }

    def __init__(self, suit_name, rank, face_up=False):
        if suit_name not in Card.SUITS:
            raise ValueError(f"Invalid suit name: {suit_name}")
        if rank not in Card.RANKS:
            raise ValueError(f"Invalid rank: {rank}")

        self.suit_name = suit_name
        self.suit = Card.SUITS[suit_name]  # symbol, e.g., '♠'
        self.rank = rank
        self.face_up = face_up

    @property
    def color(self):
        return Card.COLORS[self.suit_name]

    def is_red(self):
        return self.color == 'red'

    def is_black(self):
        return self.color == 'black'

    @property
    def value(self):
        """Return numerical value for comparing order (1 = A, 13 = K)"""
        return Card.RANKS.index(self.rank) + 1

    def flip(self):
        self.face_up = not self.face_up

    def __repr__(self):
        return f"[{self.rank}{self.suit}]" if self.face_up else "[XX]"

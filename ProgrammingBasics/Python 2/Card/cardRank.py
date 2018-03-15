
class Card:
    def __init__(self,suit_id, rank_id):
        self.rank_id = rank_id
        self.suit_id = suit_id

        if self.rank_id == 1:
            self.rank = "Ace"
            self.value = 1
        elif self.rank_id == 11:
            self.rank = "Jack"
            self.value = 11
        elif self.rank_id == 12:
            self.rank = "Queen"
            self.value = 12
        elif self.rank_id == 13:
            self.rank = "King"
            self.value = 1
        elif 2 <= self.rank_id <= 10:
            self.rank = str(self.rank_id)
            self.value = self.rank_id
        else:
            self.rank = "Rank error"
            self.value =- 1

        if self.suit_id == 1:
            self.suit = "DIAMONDS"
        elif self.suit_id == 2:
            self.suit = "HEARTS"
        elif self.suit_id == 3:
            self.suit = "SPADES"
        elif self.suit_id == 4:
            self.suit = "CLUBS"
        else:
            suit_id = "SuitError"
        self.short_name = self.rank[0] + self.suit[0]
        if self.rank == "10":
            self.short_name = self.rank + self.suit[0]
        self.long_name = self.rank + " of " + self.suit


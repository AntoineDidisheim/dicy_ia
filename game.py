import numpy as np


class Game:
    def __init__(self, blind=1, verbose=False):
        self.current_bet = 0
        self.blind = blind
        self.bet = blind * 2
        self.pot = 0
        self.dead_hand = -1
        self.dice = 1
        self.verbose = verbose
        self.card_played = []

    def start_round(self):
        self.pot = self.blind
        self.card_played = []
        # now we shuffle the deck
        deck = np.append(np.linspace(1, 12, 12), np.linspace(1, 12, 12))
        deck = np.append(np.linspace(1, 12, 12), deck)
        hands = np.random.choice(deck, size=(12, 3), replace=False)
        self.dead_hand = hands[:, 2]
        hands = hands[:, :2]
        return hands

    def dice_roll(self):
        self.dice = np.random.randint(1, 6) + np.random.randint(1, 6)
        if self.verbose:
            print("Dice roll: ", self.dice)

    def get_dead_cards(self, nb_dead_cards=3, delete_given_cards=True):
        choice = np.random.choice(self.dead_hand, size=nb_dead_cards, replace=False)
        if delete_given_cards:
            self.dead_hand = self.del_cards_from_array(choice, self.dead_hand)
        return choice

    def del_cards_from_array(self, array_to_del, from_this_array):
        for a in array_to_del:
            ind = np.where(from_this_array == a)[0][0]
            from_this_array = np.delete(arr=from_this_array, obj=ind)
        return from_this_array

    def get_winner(self):
        assert len(self.card_played) == 2, "problem with number of card played: " + self.card_played
        S = []
        for card in self.card_played:
            s = card[0]
            if s > self.dice:
                s = s/100
            if s == 1:
                s = 0
            S.append(s)
        S = np.array(S)
        if S[0]==S[1]:
            res = -1
        else:
            res = np.argmax(S)
        return res

    def end_of_phase(self):
        self.card_played = []


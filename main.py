from game import Game
from players import PlayerNNet
import numpy as np
import random

nb_players = 2
players = []
for i in range(nb_players):
    pl = PlayerNNet(id=i)
    players.append(pl)


for full_game_id in range(5):
    # start the round
    game = Game(verbose=False)
    hands = game.start_round()

    for i in range(hands.shape[1]):
        players[i].get_hand(hands[:, i])
        new_card = game.get_dead_cards()
        players[i].update_hand(new_card)

    for p in range(12):
        # define who it is to play
        first_player = (p % 2 == 0) * 1
        pl1 = players[first_player]
        pl2 = players[1 - first_player]
        #  start the pass
        game.dice_roll()
        for i in range(nb_players):
            game.card_played.append(players[i].decide_card_to_play(game.dice))

        # now first player decide if he bet
        bet = pl1.decide_to_bet()
        follow = "not relevant"
        res = "not_relevant"
        if not bet:
            # the first player did not bet game over right now, update money and learning on bet
            if game.card_played[1-first_player] == 1:  # check if second player card is the troll
                pl1.add_to_score(-4)
                pl1.update_betting_strategies(-4)
                pl2.add_to_score(4)
            else:
                pl1.add_to_score(-1)
                pl1.update_betting_strategies(-1)
                pl2.add_to_score(1)
        else:
            follow = pl2.decide_to_follow()
            if not follow:
                # here player 2 fold so we update his following and player 1 betting
                if game.card_played[1-first_player] == 1:
                    pl1.add_to_score(4)
                    pl1.update_betting_strategies(4)
                    pl2.add_to_score(-4)
                    pl2.update_following_strategies(-4)
                else:
                    pl1.add_to_score(1)
                    pl1.update_betting_strategies(1)
                    pl2.add_to_score(-1)
                    pl2.update_following_strategies(-1)
            else:
                # here we have a followed bet. The game will be resolved through score
                # first we add the revealed card to known history
                for i in range(nb_players):
                    players[i].learn_new_card(game.card_played[1 - i])
                # now we see who wins
                res = game.get_winner()
                if res == -1:
                    # draw
                    pl1.add_to_score(0)
                    pl1.update_betting_strategies(0)
                    pl2.add_to_score(0)
                    pl2.update_betting_strategies(0)
                else:
                    # we have a winner and res is the index
                    if res != first_player:
                        #pl1 wins!
                        pl1.add_to_score(2)
                        pl1.update_betting_strategies(2)
                        pl2.add_to_score(-2)
                        pl2.update_following_strategies(-2)
                    else:
                        #pl2 wins!
                        pl1.add_to_score(-2)
                        pl1.update_betting_strategies(-2)
                        pl2.add_to_score(2)
                        pl2.update_following_strategies(-2)
        # end of p
        game.end_of_phase()


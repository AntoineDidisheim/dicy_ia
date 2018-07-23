from game import Game
from players import PlayerNNetWithExternalBrain
import numpy as np
import random
from brain import Brain
import time

# creating the shared brain
brain = Brain()

# creating the players with shared brain
nb_players = 2
players = []
for i in range(nb_players):
    pl = PlayerNNetWithExternalBrain(id=i, brain=brain)  # notice that the same instance is used for the brain
    players.append(pl)

t = time.time()
# start the loop of number of game played in simulation
for full_game_id in range(10000000000000000000):
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

        card1 = game.card_played[first_player]
        card2= game.card_played[1-first_player]


        # now first player decide if he bet
        bet = pl1.decide_to_bet()
        follow = "not relevant"
        res = "not_relevant"
        if not bet:
            # the first player did not bet game over right now, update money and learning on bet
            # whatever is the true win we update the bluff strategy as should have bluffed
            if 1 in pl2.hand or card2 ==1:
                pl2.update_bluff_strategies(4)
            if card2 == 1:  # check if second player card is the troll
                pl1.add_to_score(-4)
                pl1.update_betting_strategies(-4)
                pl2.add_to_score(4)
            else:
                pl1.add_to_score(-1)
                pl1.update_betting_strategies(-1)
                pl2.add_to_score(1)
        else:
            # since pl1 decided to bet, pl2 shouldn't have bluff so we learn
            if card2 == 1:
                pl2.update_bluff_strategies(-2)

            follow = pl2.decide_to_follow()
            if not follow:
                # learn that should have bluffed if possible for pl 1 since pl2 folded
                if 1 in pl1.hand or card1 == 1:
                    pl1.update_bluff_strategies(4)
                # here player 2 fold so we update his following and player 1 betting
                if card1 == 1:
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
                # learn that should not have bluffed if followed as pl2 followed
                if card1 == 1:
                    pl1.update_bluff_strategies(-2)
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
                    if res == first_player:
                        # pl1 wins!
                        pl1.add_to_score(2)
                        pl1.update_betting_strategies(2)
                        pl2.add_to_score(-2)
                        pl2.update_following_strategies(-2)
                    else:
                        # pl2 wins!
                        pl1.add_to_score(-2)
                        pl1.update_betting_strategies(-2)
                        pl2.add_to_score(2)
                        pl2.update_following_strategies(-2)
        # end of p
        game.end_of_phase()
    # end of game
    if full_game_id%1000==0:
        print('Game ',full_game_id, 'finished')
    if full_game_id%10000==0:
        brain.new_path_file(game_id=full_game_id)
        brain.save_all_nets()
        print("save brain progress")


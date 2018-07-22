from game import Game
from players import PlayerNNetWithExternalBrain
import numpy as np
import random
from brain import Brain
import time

# creating the individual brain
brain_trained = Brain(id=0)

brain_trained.new_path_file(290000)
brain_trained.restore_all_nets()
brain_new = Brain(id="untrained")

# creating the players with shared brain
nb_players = 2
players = [PlayerNNetWithExternalBrain(brain=brain_trained,id='trained'),
           PlayerNNetWithExternalBrain(brain=brain_new,id='untrained')]

# the second player has the bad brain and a full random behavior
players[0].epsilon=0
players[1].epsilon=0

# the rest is similar to training except player are different and I disabeled learning
score_player_0 = []
score_player_1 = []
players_follow = [[], []]
players_bet = [[], []]

t = time.time()
# start the loop of number of game played in simulation
for full_game_id in range(500):
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
        card2 = game.card_played[1-first_player]


        # now first player decide if he bet
        bet = pl1.decide_to_bet()
        players_bet[first_player].append(bet)
        follow = "not relevant"
        res = "not_relevant"
        if not bet:
            # the first player did not bet game over right now, update money and learning on bet
            if card2 == 1:  # check if second player card is the troll
                pl1.add_to_score(-4)
                pl1.update_betting_strategies(-4)
                pl2.add_to_score(4)
            else:
                pl1.add_to_score(-1)
                pl1.update_betting_strategies(-1)
                pl2.add_to_score(1)
        else:
            follow = pl2.decide_to_follow()
            players_follow[1-first_player].append(follow)
            if not follow:
                # here player 2 fold so we update his following and player 1 betting
                if card1 == 1:
                    pl1.add_to_score(4)
                    # pl1.update_betting_strategies(4)
                    pl2.add_to_score(-4)
                    # pl2.update_following_strategies(-4)
                else:
                    pl1.add_to_score(1)
                    # pl1.update_betting_strategies(1)
                    pl2.add_to_score(-1)
                    # pl2.update_following_strategies(-1)
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
                    # pl1.update_betting_strategies(0)
                    pl2.add_to_score(0)
                    # pl2.update_betting_strategies(0)
                else:
                    # we have a winner and res is the index
                    if res == first_player:
                        # pl1 wins!
                        pl1.add_to_score(2)
                        # pl1.update_betting_strategies(2)
                        pl2.add_to_score(-2)
                        # pl2.update_following_strategies(-2)
                    else:
                        # pl2 wins!
                        pl1.add_to_score(-2)
                        # pl1.update_betting_strategies(-2)
                        pl2.add_to_score(2)
                        # pl2.update_following_strategies(-2)
        game.end_of_phase()
    if full_game_id%1000==0:
        print('Game ',full_game_id, 'finished')
    score_player_0.append(players[0].current_score)
    score_player_1.append(players[1].current_score)


from matplotlib import pyplot as plt
plt.plot(score_player_0)
plt.show()


print('mean bet player trained:', np.mean(players_bet[0]))
print('mean bet player UNTRAINED:', np.mean(players_bet[1]))
print('mean follow player trained:', np.mean(players_follow[0]))
print('mean follow player UNTRAINED:', np.mean(players_follow[1]))
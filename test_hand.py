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
for full_game_id in range(1):
    # start the round
    game = Game(verbose=False)
    hands = game.start_round()
    print('-----------',full_game_id ,'------------')
    for i in range(hands.shape[1]):
        players[i].get_hand(hands[:, i])
        new_card = game.get_dead_cards()
        players[i].update_hand(new_card)

    for p in range(12):
        # define who it is to play
        first_player = (p % 2 == 0) * 1
        pl1 = players[first_player]
        pl2 = players[1 - first_player]

        print('###', p, '###')
        print("Hand:", np.sort(players[0].hand))

        #  start the pass
        game.dice_roll()
        for i in range(nb_players):
            game.card_played.append(players[i].decide_card_to_play(game.dice))

        card1 = game.card_played[0]
        card2 = game.card_played[1]

        print("Dice: ",game.dice)
        print("Card choice",card1)

        game.end_of_phase()


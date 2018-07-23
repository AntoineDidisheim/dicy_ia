import numpy as np
import random
from game import Game
import tensorflow as tf
import os


class Player:
    def __init__(self, epsilon=0.1):
        self.current_score = 0
        self.hand = np.linspace(1, 12, 12)
        self.history_known_cards = np.linspace(1, 12, 12)
        self.dice = 1
        self.epsilon = epsilon
        self.current_card = 1

    @staticmethod
    def del_cards_from_array(array_to_del, from_this_array):
        for a in array_to_del:
            ind = np.where(from_this_array == a)[0][0]
            from_this_array = np.delete(arr=from_this_array, obj=ind)
        return from_this_array

    def update_hand(self, new_cards):
        # currently simple random choice
        choice = np.random.choice(self.hand, size=3, replace=False)
        self.history_known_cards = np.append(self.history_known_cards, new_cards)
        for a in choice:
            ind = np.where(self.hand == a)[0][0]
            self.hand = np.delete(arr=self.hand, obj=ind)
        self.hand = np.append(self.hand, new_cards)

        assert len(self.hand) == 12, "problem with update_hand method"
        return choice

    def get_hand(self, new_hand):
        self.hand = new_hand
        self.history_known_cards = new_hand

    def learn_new_card(self, card):
        self.history_known_cards = np.append(self.history_known_cards, card)

    def process_card_count(self, cards):
        res = []
        for i in range(1, 13):
            count = (np.sum(cards == i) - 1.5) / 1.5  # centered and standardized
            res.append(count)
        return res

    def create_bet_or_follow_input(self, bet=True):
        res = np.append(self.process_card_count(self.history_known_cards), self.create_score(self.current_card))
        res = np.append(res, (self.dice - 3.5) / 3.5)

        # for i in range(1, 13):
        #     res = np.append(res, self.create_score(i))

        res = res.reshape((1, len(res)))
        return res

    def create_card_choice_input(self):
        res = np.append(self.process_card_count(self.history_known_cards),
                        self.process_card_count(self.hand))
        res = np.append(res, (self.dice - 3.5) / 3.5)

        # for i in range(1, 13):
        #     res = np.append(res, self.create_score(i))

        res = res.reshape((1, len(res)))
        return res

    def create_bluff_choice_input(self):
        res = np.append(self.process_card_count(self.history_known_cards),
                        self.process_card_count(self.hand))
        res = np.append(res, (self.dice - 3.5) / 3.5)

        # for i in range(1, 13):
        #     res = np.append(res, self.create_score(i))

        res = res.reshape((1, len(res)))
        return res

    def create_score(self, value):
        score = 0
        if value > 1:
            if value <= self.dice:
                score = value
            else:
                score = value / 10
        return score

    def decide_to_bet(self):
        # again here it is random where it will be decisions
        if random.random() <= self.epsilon:
            bet = random.random() > 0.5
        else:
            bet = self.high_choice_of_bet()
        return bet

    def decide_to_follow(self):
        # again here it is random where it will be decisions
        if random.random() <= self.epsilon:
            follow = random.random() > 0.5
        else:
            follow = self.high_choice_of_follow()
        return follow

    def decide_card_to_play(self, dice):
        self.dice = dice  # we update the dice at this stage
        # now random but it will be function of the dice roll

        # if random.random() <= 1:
        if random.random() <= self.epsilon:
            choice = np.random.choice(self.hand, size=1)
        else:
            choice = self.high_choice_of_card()

        self.hand = self.del_cards_from_array(array_to_del=np.array([choice]),
                                              from_this_array=self.hand)
        self.current_card = choice
        return choice

    def add_to_score(self, value_to_add):
        self.current_score = self.current_score + value_to_add

    def update_betting_strategies(self, reward):
        self.update_card_strategy(reward)

    def update_following_strategies(self, reward):
        self.update_card_strategy(reward)

    def update_card_strategy(self, reward):
        pass

    def high_choice_of_follow(self):
        return True

    def high_choice_of_bet(self):
        return True

    def high_choice_of_card(self):
        return 1


class PlayerNNet(Player):
    def __init__(self, epsilon=0.1, id=0):
        super().__init__(epsilon=epsilon)

        self.id = "player" + str(id)

        # creating the saving path and, if need be, directories
        self.path_card = 'saved_model/' + str(self.id) + '/card_model/'
        if not os.path.exists(self.path_card):
            os.makedirs(self.path_card)
        self.path_bet = 'saved_model/' + str(self.id) + '/bet_model/'
        if not os.path.exists(self.path_bet):
            os.makedirs(self.path_bet)
        self.path_follow = 'saved_model/' + str(self.id) + '/follow_model/'
        if not os.path.exists(self.path_follow):
            os.makedirs(self.path_follow)

        # defining the bet nnet
        self.graph_bet = tf.Graph()
        with self.graph_bet.as_default() as g:
            self.x_bet, self.y_bet, self.q_bet, _ = self.define_two_layer_nnet(
                input_dim=max(self.create_bet_or_follow_input().shape),
                output_dim=1)
            loss = tf.square(self.y_bet - self.q_bet)  # keeping square here include risk aversion
            self.train_op_bet = tf.train.AdagradOptimizer(0.01).minimize(loss)
            self.sess_bet = tf.Session(graph=g)
            self.sess_bet.run(tf.global_variables_initializer())
            self.saver_bet = tf.train.Saver()

        # defining the follow nnet
        self.graph_follow = tf.Graph()
        with self.graph_follow.as_default() as g:
            self.x_follow, self.y_follow, self.q_follow, _ = self.define_two_layer_nnet(
                input_dim=max(self.create_bet_or_follow_input().shape),
                output_dim=1)
            loss = tf.square(self.y_follow - self.q_follow)  # keeping square here include risk aversion
            self.train_op_follow = tf.train.AdagradOptimizer(0.01).minimize(loss)
            self.sess_follow = tf.Session(graph=g)
            self.sess_follow.run(tf.global_variables_initializer())
            self.saver_follow = tf.train.Saver()

        # defining the card nnet
        self.graph_card = tf.Graph()
        with self.graph_card.as_default() as g:
            self.x_card, self.y_card, self.q_card, self.testW = self.define_two_layer_nnet(
                input_dim=max(self.create_card_choice_input().shape),
                output_dim=12)
            loss = tf.square(self.y_card - self.q_card)  # keeping square here include risk aversion
            self.train_op_card = tf.train.AdagradOptimizer(0.01).minimize(loss)
            self.sess_card = tf.Session(graph=g)
            self.sess_card.run(tf.global_variables_initializer())
            self.saver_card = tf.train.Saver()

    def define_two_layer_nnet(self, input_dim, output_dim, h1_dim=200):
        x = tf.placeholder(tf.float32, [None, input_dim])
        y = tf.placeholder(tf.float32, [None, output_dim])
        w1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        w2 = tf.Variable(tf.random_normal([h1_dim, h1_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        w3 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
        b3 = tf.Variable(tf.constant(0.01, shape=[output_dim]))
        q = tf.add(tf.matmul(h2, w3), b3)
        return x, y, q, w3

    def update_betting_strategies(self, reward):
        self.update_card_strategy(reward)
        reward = np.array([reward]).reshape((1, 1))
        self.sess_bet.run(self.train_op_bet,
                          feed_dict={self.x_bet: self.create_bet_or_follow_input(), self.y_bet: reward})

    def update_following_strategies(self, reward):
        self.update_card_strategy(reward)
        reward = np.array([reward]).reshape((1, 1))
        self.sess_follow.run(self.train_op_follow,
                             feed_dict={self.x_follow: self.create_bet_or_follow_input(), self.y_follow: reward})

    def update_card_strategy(self, reward):
        y = np.zeros(shape=(1, 12))
        y[:, int(self.current_card - 1)] = reward
        self.sess_card.run(self.train_op_card,
                           feed_dict={self.x_card: self.create_card_choice_input(), self.y_card: y})

    def high_choice_of_follow(self, get_bool_answer=True):
        predicted_profits = self.sess_follow.run(self.q_follow,
                                                 feed_dict={self.x_follow: self.create_bet_or_follow_input()})
        if get_bool_answer:
            res = predicted_profits > 0
        else:
            res = predicted_profits
        return res

    def high_choice_of_bet(self, get_bool_answer=True):
        predicted_profits = self.sess_bet.run(self.q_bet, feed_dict={self.x_bet: self.create_bet_or_follow_input()})
        if get_bool_answer:
            res = predicted_profits > 0
        else:
            res = predicted_profits
        return res

    def high_choice_of_card(self):
        predicted_profits = self.sess_card.run(self.q_card, feed_dict={self.x_card: self.create_card_choice_input()})
        predicted_profits = predicted_profits.reshape(12)
        # we have a predicted profits for all potential cards
        # first we create a list of index of cards that we can play
        ind = np.unique(self.hand) - 1
        # inside the subsample of cards we want to select the highest predicted value
        ind_in_ind = np.argmax(predicted_profits[ind.astype(int)])
        # back to the ind we just add 1 to move again from index to card now
        card = ind[ind_in_ind] + 1
        card = np.array([card])
        return card

    def save_all_nets(self):
        self.saver_card.save(sess=self.sess_card, save_path=self.path_card + 'm.ckpt')
        self.saver_follow.save(sess=self.sess_follow, save_path=self.path_follow + 'm.ckpt')
        self.saver_bet.save(sess=self.sess_bet, save_path=self.path_bet + 'm.ckpt')

    def restore_all_nets(self):
        with self.graph_card.as_default() as g:
            self.saver_card.restore(sess=self.sess_card, save_path=self.path_card + 'm.ckpt')
        with self.graph_follow.as_default() as g:
            self.saver_follow.restore(sess=self.sess_follow, save_path=self.path_follow + 'm.ckpt')
        with self.graph_bet.as_default() as g:
            self.saver_bet.restore(sess=self.sess_bet, save_path=self.path_bet + 'm.ckpt')

        # print(self.sess_card.run(self.testW))


class PlayerNNetWithExternalBrain(Player):
    def __init__(self, brain, epsilon=0.1, id=0, yes_man = False):
        super().__init__(epsilon=epsilon)
        self.id = "player" + str(id)
        self.brain = brain
        self.yes_man = yes_man

    def update_betting_strategies(self, reward):
        # self.update_card_strategy(reward)
        reward = np.array([reward]).reshape((1, 1))
        self.brain.sess_bet.run(self.brain.train_op_bet,
                                feed_dict={self.brain.x_bet: self.create_bet_or_follow_input(),
                                           self.brain.y_bet: reward})

    def update_following_strategies(self, reward):
        # self.update_card_strategy(reward)
        reward = np.array([reward]).reshape((1, 1))
        self.brain.sess_follow.run(self.brain.train_op_follow,
                                   feed_dict={self.brain.x_follow: self.create_bet_or_follow_input(),
                                              self.brain.y_follow: reward})

    def update_bluff_strategies(self, reward):
        # self.update_card_strategy(reward)
        reward = np.array([reward]).reshape((1, 1))
        self.brain.sess_bluff.run(self.brain.train_op_bluff,
                                  feed_dict={self.brain.x_bluff: self.create_bluff_choice_input(),
                                             self.brain.y_bluff: reward})

    def update_card_strategy(self, reward):
        y = np.zeros(shape=(1, 12))
        y[:, int(self.current_card - 1)] = reward
        self.brain.sess_card.run(self.brain.train_op_card,
                                 feed_dict={self.brain.x_card: self.create_card_choice_input(),
                                            self.brain.y_card: y})

    def high_choice_of_follow(self, get_bool_answer=True):
        predicted_profits = self.brain.sess_follow.run(self.brain.q_follow,
                                                       feed_dict={
                                                           self.brain.x_follow: self.create_bet_or_follow_input()})
        if get_bool_answer:
            res = predicted_profits >= 0
        else:
            res = predicted_profits

        if self.yes_man:
            res = True

        return res

    def high_choice_of_bet(self, get_bool_answer=True):
        predicted_profits = self.brain.sess_bet.run(self.brain.q_bet,
                                                    feed_dict={self.brain.x_bet: self.create_bet_or_follow_input()})
        if get_bool_answer:
            res = predicted_profits >= 0
        else:
            res = predicted_profits

        if self.yes_man:
            res = True

        return res

    def high_choice_bluff(self, get_bool_answer=True):
        predicted_profits = self.brain.sess_bluff.run(self.brain.q_bluff,
                                                      feed_dict={self.brain.x_bluff: self.create_bluff_choice_input()})
        if get_bool_answer:
            res = predicted_profits >= 0
        else:
            res = predicted_profits

        if self.yes_man:
            res = True

        return res

    def high_choice_of_card(self):
        predicted_profits = self.brain.sess_card.run(self.brain.q_card,
                                                     feed_dict={self.brain.x_card: self.create_card_choice_input()})
        predicted_profits = predicted_profits.reshape(12)
        # we have a predicted profits for all potential cards
        # first we create a list of index of cards that we can play
        ind = np.unique(self.hand) - 1
        # inside the subsample of cards we want to select the highest predicted value
        ind_in_ind = np.argmax(predicted_profits[ind.astype(int)])
        # back to the ind we just add 1 to move again from index to card now
        card = ind[ind_in_ind] + 1
        card = np.array([card])
        return card

        # print(self.sess_card.run(self.testW))

    def decide_card_to_play(self, dice):
        self.dice = dice  # we update the dice at this stage
        # now random but it will be function of the dice roll

        m_score = -10000
        choice = 'not_selected'
        for c in self.hand:
            s = self.create_score(c)
            if s > m_score:
                m_score = s
                choice = c

        if 1 in self.hand:
            # we can and maybe should bluff
            if random.random() <= self.epsilon:
                bluff = random.random() < 0.5
            else:
                bluff = self.high_choice_bluff()
            if bluff:
                choice = 1

        # finally we decide if or not we do change the card
        self.hand = self.del_cards_from_array(array_to_del=np.array([choice]),
                                              from_this_array=self.hand)
        self.current_card = choice
        return np.array([choice])

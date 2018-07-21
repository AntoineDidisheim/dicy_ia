import tensorflow as tf
import numpy as np
import os
from players import PlayerNNet


class Brain():
    def __init__(self, id=0):
        self.id = "brain"+str(id)

        # creating the saving path and, if need be, directories
        self.path_card = 'saved_model/' + str(self.id) + '/card_model/'
        self.path_bet = 'saved_model/' + str(self.id) + '/bet_model/'
        self.path_follow = 'saved_model/' + str(self.id) + '/follow_model/'
        self.create_path_file()

        # we create a temp player just to get the correct dimension we will need for net in and out
        playerTemp = PlayerNNet()

        # defining the bet nnet
        self.graph_bet = tf.Graph()
        with self.graph_bet.as_default() as g:
            self.x_bet, self.y_bet, self.q_bet, _ = self.define_two_layer_nnet(
                input_dim=max(playerTemp.create_bet_or_follow_input().shape),
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
                input_dim=max(playerTemp.create_bet_or_follow_input().shape),
                output_dim=1)
            loss = tf.square(self.y_follow - self.q_follow)  # keeping square here include risk aversion
            self.train_op_follow = tf.train.AdagradOptimizer(0.01).minimize(loss)
            self.sess_follow = tf.Session(graph=g)
            self.sess_follow.run(tf.global_variables_initializer())
            self.saver_follow = tf.train.Saver()

        # defining the card nnet
        self.graph_card = tf.Graph()
        with self.graph_card.as_default() as g:
            self.x_card, self.y_card, self.q_card, self.testW= self.define_two_layer_nnet(
                input_dim=max(playerTemp.create_card_choice_input().shape),
                output_dim=12)
            loss = tf.square(self.y_card - self.q_card)  # keeping square here include risk aversion
            self.train_op_card = tf.train.AdagradOptimizer(0.01).minimize(loss)
            self.sess_card = tf.Session(graph=g)
            self.sess_card.run(tf.global_variables_initializer())
            self.saver_card = tf.train.Saver()

    def define_two_layer_nnet(self, input_dim, output_dim, h1_dim=500):
        x = tf.placeholder(tf.float32, [None, input_dim])
        y = tf.placeholder(tf.float32, [None, output_dim])
        w1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        w2 = tf.Variable(tf.random_normal([h1_dim, h1_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        w3 = tf.Variable(tf.random_normal([h1_dim, h1_dim]))
        b3 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

        w4 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
        b4 = tf.Variable(tf.constant(0.01, shape=[output_dim]))
        q = tf.add(tf.matmul(h3, w4), b4)
        return x, y, q, w3

    def new_path_file(self,game_id):
        self.path_card = 'saved_model/' + str(self.id) + '/game_'+str(game_id)+'/card_model/'
        self.path_bet = 'saved_model/' + str(self.id) + '/game_'+str(game_id)+'/bet_model/'
        self.path_follow = 'saved_model/' + str(self.id) + '/game_'+str(game_id)+'/follow_model/'
        self.create_path_file()

    def create_path_file(self):
        if not os.path.exists(self.path_bet):
            os.makedirs(self.path_bet)
        if not os.path.exists(self.path_card):
            os.makedirs(self.path_card)
        if not os.path.exists(self.path_follow):
            os.makedirs(self.path_follow)

    def save_all_nets(self):
        self.saver_card.save(sess=self.sess_card, save_path=self.path_card+'m.ckpt')
        self.saver_follow.save(sess=self.sess_follow, save_path=self.path_follow+'m.ckpt')
        self.saver_bet.save(sess=self.sess_bet, save_path=self.path_bet+'m.ckpt')

    def restore_all_nets(self):
        with self.graph_card.as_default() as g:
            self.saver_card.restore(sess=self.sess_card,save_path=self.path_card+'m.ckpt')
        with self.graph_follow.as_default() as g:
            self.saver_follow.restore(sess=self.sess_follow, save_path=self.path_follow+'m.ckpt')
        with self.graph_bet.as_default() as g:
            self.saver_bet.restore(sess=self.sess_bet, save_path=self.path_bet+'m.ckpt')


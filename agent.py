import os
import json
import copy
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from keras.src.optimizers import Adam
from model import Shared_Model
import tensorflow as tf

import matplotlib.pyplot as plt

class CustomAgent:
    # A custom Bitcoin trading agent
    def __init__(self, lookback_window_size=50, lr=0.00005, epochs=1, stocks=[], optimizer=Adam, batch_size=32, model='', shape = [],depth=0, comment=""):
        self.lookback_window_size = lookback_window_size
        self.comment = comment
        self.depth = depth
        self.stocks = stocks
        self.shape = shape
        self.model = model

        # Action Space it goes from 0 to the number of assets in the portfolio
        self.action_space = np.array(range(0,len(self.stocks)))

        # Create a folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"

        # State size contains Market+Orders+Indicators history for the last lookback_window_size steps
        if self.model =="EIIE":
            self.state_size = (len(stocks), lookback_window_size, self.shape[1])
        else:
            self.state_size = (lookback_window_size, self.shape[1]*self.shape[2]+2+2*self.shape[2]) # 5 standard OHCL information + market and indicators

        # Neural Networks part
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer, model=self.model)



    # create tensorboard writer
    def create_writer(self, initial_balance, normalize_value, train_episodes):
        self.replay_count = 0
        self.writer = SummaryWriter('runs/'+self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
          os.makedirs(self.log_name)

        self.start_training_log(initial_balance, normalize_value, train_episodes)

    def start_training_log(self, initial_balance, normalize_value, train_episodes):
        # save training parameters to Parameters.json file for future
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        params = {
          "training start": current_date,
          "initial balance": initial_balance,
          "training episodes": train_episodes,
          "lookback window size": self.lookback_window_size,
          "depth": self.depth,
          "lr": self.lr,
          "epochs": self.epochs,
          "batch size": self.batch_size,
          "normalize value": normalize_value,
          "model": self.model,
          "comment": self.comment,
          "saving time": "",
          "Actor name": "",
          "Critic name": "",
        }
        with open(self.log_name+"/Parameters.json", "w") as write_file:
          json.dump(params, write_file, indent=4)


    @tf.function
    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)
        values = tf.cast(values, tf.float32)
        next_values = tf.cast(next_values, tf.float32)
        
        deltas = rewards + gamma * (1.0 - dones) * next_values - values
        gaes = tf.identity(deltas)
        
        for t in tf.range(tf.shape(deltas)[0] - 2, -1, -1):
            gaes = tf.tensor_scatter_nd_update(
                gaes,
                [[t]],
                [gaes[t] + (1.0 - dones[t]) * gamma * lamda * gaes[t + 1]]
            )
        
        target = gaes + values
        if normalize:
            gaes = (gaes - tf.reduce_mean(gaes)) / (tf.math.reduce_std(gaes) + 1e-8)
        return gaes, target

    def replay(self, states, orders, rewards, predictions, dones, next_states, orders_history):
        # Convert inputs to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        orders = tf.convert_to_tensor(orders, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        orders_history = tf.convert_to_tensor(orders_history, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)

        if self.model == "EIIE":
            values = self.Critic.critic_predict(states, tf.expand_dims(orders, axis=1))
            next_values = self.Critic.critic_predict(next_states, tf.expand_dims(orders_history, axis=1))
        else:
            values = self.Critic.critic_predict(states, tf.expand_dims(tf.expand_dims(orders, axis=0), axis=0))
            next_values = self.Critic.critic_predict(next_states, tf.expand_dims(tf.expand_dims(orders_history, axis=0), axis=0))

        advantages, target = self.get_gaes(rewards, dones, tf.squeeze(values), tf.squeeze(next_values))
        y_true = tf.concat([advantages, predictions], axis=1)

        # Stack everything to numpy array
        y_true = np.hstack([advantages, predictions])


        # training Actor and Critic networks
        if self.model == "EIIE":
            a_loss = self.Actor.Actor.fit([states,np.expand_dims(orders, axis=1)], y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
            c_loss = self.Critic.Critic.fit([states,np.expand_dims(orders, axis=1)], target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        else:
            a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
            c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    @tf.function
    def act(self, state, order):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        order = tf.convert_to_tensor(order, dtype=tf.float32)
        
        if self.model == "EIIE":
            # Reshape order to match expected input shape
            if len(order.shape) == 2:
                order = tf.expand_dims(tf.expand_dims(order, axis=0), axis=1)
            elif len(order.shape) == 3:
                order = tf.expand_dims(order, axis=1)
        else:
            state = tf.expand_dims(state, axis=0)
            order = tf.expand_dims(tf.expand_dims(order, axis=0), axis=0)
            
        prediction = self.Actor.actor_predict(state, order)
        return prediction[0]

    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{self.log_name}/{score}_{name}_Actor.weights.h5")
        self.Critic.Critic.save_weights(f"{self.log_name}/{score}_{name}_Critic.weights.h5")

        # update json file settings
        if score != "":
          with open(self.log_name+"/Parameters.json", "r") as json_file:
            params = json.load(json_file)
          params["saving time"] = datetime.now().strftime('%Y-%m-%d %H:%M')
          params["Actor name"] = f"{score}_{name}_Actor.h5"
          params["Critic name"] = f"{score}_{name}_Critic.h5"
          with open(self.log_name+"/Parameters.json", "w") as write_file:
            json.dump(params, write_file, indent=4)

        # log saved model arguments to file
        if len(args) > 0:
          with open(f"{self.log_name}/log.txt", "a+") as log:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            arguments = ""
            for arg in args:
              arguments += f", {arg}"
            log.write(f"{current_time}{arguments}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(os.path.join(folder, f"{name}_Critic.h5"))
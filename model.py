import numpy as np
from keras import Model
from keras.src.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Conv2D, Activation, ZeroPadding2D, Concatenate
from keras import backend as K




class Shared_Model:
    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        X_input = Input(input_shape)
        self.action_space = action_space

        self.model = model

        # Shared CNN layers:
        if model=="CNN":
          X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
          X = MaxPooling1D(pool_size=2)(X)
          X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
          X = MaxPooling1D(pool_size=2)(X)
          X = Flatten()(X)
        #EIIE Layers
        elif model=="EIIE":
          X = Conv2D(2, (3, 1))(X_input)
          X = Activation('relu')(X)
          X = Conv2D(20, (48, 1))(X)
          X = Activation('relu')(X)
          # print("X shape:", X.shape)

          inputB = Input(shape=(1, 50, 10))
          modelB = Conv2D(filters=2, kernel_size=(3, 1), activation='relu')(inputB)
          modelB = Conv2D(filters=20, kernel_size=(50 - 2, 1), activation='relu')(modelB)
          modelB = ZeroPadding2D(padding=((0, 0), (0, 4)))(modelB)
          # print("modelB shape:", modelB.shape)
          merged = Concatenate(axis=3)([X, modelB])
          X = Conv2D(filters=1, kernel_size=(1, 1))(merged)

          #output = Dense(self.action_space, activation="softmax")(x)
        # Shared LSTM layers:
        elif model=="LSTM":
          X = LSTM(512, return_sequences=True)(X_input)
          X = LSTM(256)(X)

        # Shared Dense layers:
        else:
          X = Flatten()(X_input)
          X = Dense(512, activation="relu")(X)

        # Critic model
        V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)
        if model == "EIIE":
          self.Critic = Model(inputs=[X_input,inputB], outputs = value)
        else:
          self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))

        # Actor model
        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)
        if model == "EIIE":
          self.Actor = Model(inputs = [X_input,inputB], outputs = output)
        else:
          self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks = y_true[:, :1], y_true[:, 1:1+self.action_space]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # NOTE: we just subtract the logs, which is the same as
        # dividing the values and then canceling the log with e^log.
        # For why we use log probabilities instead of actual probabilities,
        # here's a great explanation:
        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        # TL;DR makes gradient ascent easier behind the scenes.
        prob = y_pred
        old_prob = prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state, order):
        if self.model == "EIIE":
          return self.Actor.predict([state, order])
        else:
          return self.Actor.predict([state, np.zeros((state.shape[0], 1))])

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state, order):
        if self.model == "EIIE":
          return self.Critic.predict([state, order])
        else:
          return self.Critic.predict([state, np.zeros((state.shape[0], 1))])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(64, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))


    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)

        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

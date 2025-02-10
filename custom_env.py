import random
import time
from collections import deque
import numpy as np
import tensorflow as tf

class CustomEnv:
    def __init__(self, df, df_normalized, initial_balance=1000, stocks=['USDCUSDT','BTCUSDT','BNBBTC','BNBBTC'],lookback_window_size=50, model=''):
        # Define action space and state size and other custom parameters
        self.xarray = df_normalized
        self.df = df
        self.df_total_steps = self.xarray.shape[0]
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size		# Historical data window
        self.normalize_value = 40000		#Value to normalize transaction data
        self.model = model

        self.weights = [1]+[0]*(self.xarray.shape[2]-1)		# Initial Weights
        self.quants = [0]*self.xarray.shape[2] 		#Initial quantities
        self.quants_ubah = [0]*self.xarray.shape[2]		# Initial quantities for buy and hold

        self.cash = 0 # Amout of cash

        self.stocks =  stocks 	# list of assets
        self.market_state = dict.fromkeys(self.stocks)		# Dict for each asset

        #Initial amount of money to Buy n hold
        self.ubah = initial_balance

        #Deque for Order History
        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)  # Market history contains the OHCL/Technical Features values for the last lookback_window_size prices (open, high, close, low)

    @tf.function
    def reset(self, env_steps_size = 0):
        # Reset the state of the environment to an initial state
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.weights = [1]+[0]*(tf.shape(self.xarray)[2]-1)
        self.quants = [0]*tf.shape(self.xarray)[2]
        self.quants_ubah = [0]*tf.shape(self.xarray)[2]
        self.short_sell = [1,1,1]
        self.cash = self.initial_balance
        self.ubah = self.initial_balance

        # Randomly selects a value contained between the initial size of the dataset and the final size minus the number of steps.
        self.start_step = tf.random.uniform(
            shape=[],
            minval=self.lookback_window_size,
            maxval=self.df_total_steps - env_steps_size + 1,
            dtype=tf.int32
        )
        self.end_step = self.start_step + env_steps_size
        self.current_step = self.start_step #Define initial Step

        #Replace deque with TensorArray for graph mode while preserving comments
        orders_history_ta = tf.TensorArray(dtype=tf.float32, size=self.lookback_window_size)
        i = tf.constant(0)
        def cond(i, ta):
            return i < self.lookback_window_size
        def body(i, ta):
            # Append the data from end t beginning
            current = self.current_step - (self.lookback_window_size - i)
            order_val = tf.concat([
                [tf.cast(self.net_worth, tf.float32) / self.normalize_value,
                 tf.cast(self.cash, tf.float32) / self.normalize_value],
                tf.cast(self.quants, tf.float32),
                tf.cast(self.weights, tf.float32)
            ], axis=0)
            ta = ta.write(i, order_val)
            return i + 1, ta
        i, orders_history_ta = tf.while_loop(cond, body, [i, orders_history_ta])
        self.orders_history = orders_history_ta.stack()

        # Replace dict of deques with TensorArrays for market states
        market_states = []
        for j in range(len(self.stocks)):
            # Create TensorArray for each stock
            market_state_ta = tf.TensorArray(dtype=tf.float32, size=self.lookback_window_size)
            
            # Fill TensorArray with historical data
            for i in range(self.lookback_window_size):
                current_step = self.current_step - (self.lookback_window_size - i - 1)
                market_state_ta = market_state_ta.write(
                    i, 
                    tf.slice(self.xarray, begin=[current_step, 0, j], size=[1, -1, 1])[0, :, 0]
                )
            market_states.append(market_state_ta.stack())
        
        # Stack all market states
        if self.model == "EIIE":
            state = tf.stack(market_states)
        else:
            state_part = tf.concat(market_states, axis=1)
            state = tf.concat([state_part, self.orders_history], axis=1)
            
        return state, self.orders_history

    @tf.function
    def _next_observation(self):
        # Get the data points for the given current_step
        start = time.time()
        # In this step, it updates the state with the most recent point that was used in 'step', for example, in Step it takes the next point after the market history, so if the market history goes to t, in the step it takes the point t+1, in the next observation it appends this point.
        market_states = []
        for j in range(len(self.stocks)):
            market_state_ta = tf.TensorArray(dtype=tf.float32, size=self.lookback_window_size)
            for i in range(self.lookback_window_size):
                current_step = self.current_step - (self.lookback_window_size - i - 1)
                market_state_ta = market_state_ta.write(
                    i,
                    self.xarray[current_step, :, j]
                )
            market_states.append(market_state_ta.stack())
        
        if self.model == "EIIE":
            obs = tf.stack(market_states)
        else:
            obs_part = tf.concat(market_states, axis=1)
            obs = tf.concat([obs_part, self.orders_history], axis=1)
        return obs

    @tf.function
    def step(self, prediction):
        # Execute one time step within the environment
        # Use to calculate the transactions fee
        prices_ant = tf.convert_to_tensor([self.df[self.current_step,2,x] for x in range(len(self.stocks))], dtype=tf.float32)
        # One step on env
        self.current_step += 1

        # Get the prices in the current step
        prices = tf.convert_to_tensor([self.df[self.current_step,2,x] for x in range(len(self.stocks))], dtype=tf.float32)
        #Calculates the balance considering the quantities purchased in the previous step, and the prices at the current time
        self.balance = tf.cast(self.cash, tf.float32) + tf.tensordot(prices[1:], tf.convert_to_tensor(self.quants[1:], dtype=tf.float32), axes=1)

        # Use to calculate the transactions fee
        quants_ant = tf.convert_to_tensor(self.quants, dtype=tf.float32)
        #Get the quantities, considering the current values and the balance of the previous transaction
        pred = tf.convert_to_tensor(prediction, dtype=tf.float32)
        prices_tensor = tf.convert_to_tensor(prices, tf.float32)
        self.quants = tf.math.divide_no_nan(self.balance * pred, prices_tensor)
        # Calculate the tax of buying and selling, 10% of the difference between quants of the periods
        # 0,1% is the binance tax source
        tax = tf.reduce_sum(tf.abs(tf.tensordot(self.quants, prices_tensor, axes=1) - tf.tensordot(quants_ant, prices_ant, axes=1))) * 0.001

        #See the value of the cash term(Stable currency, in the future consider whether this approach is valid)
        self.cash = self.quants[0] * prices_tensor[0]
        #Save the previous net worth
        self.prev_net_worth = self.net_worth

        #Calculate the new portfolio value
        self.net_worth = tf.tensordot(self.quants, prices_tensor, axes=1) - tax

        #Append the transactions values to deque
        orders_history_ta = tf.TensorArray(dtype=tf.float32, size=1)
        order_val = tf.concat([
            [self.net_worth / self.normalize_value, self.cash / self.normalize_value],
            tf.cast(self.quants, tf.float32) / self.normalize_value,
            tf.cast(pred, tf.float32)
        ], axis=0)
        orders_history_ta = orders_history_ta.write(0, order_val)
        self.orders_history = orders_history_ta.stack()

        # Calculate reward
        reward = tf.math.log(self.net_worth / tf.cast(self.prev_net_worth, tf.float32))
        #reward = self.net_worth - self.prev_net_worthh

        if self.net_worth <= self.initial_balance / 2:
          done = True
        else:
          done = False
        obs = self._next_observation()

        return obs, self.orders_history, reward, done, prices

    def render(self):
        # render environment
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
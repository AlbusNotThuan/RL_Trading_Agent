import pandas as pd
import numpy as np
from custom_env import CustomEnv
from agent import CustomAgent
from setup import train_agent, test_agent
from keras.src.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K

#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
tf.keras.utils.disable_interactive_logging()
np.random.seed(32)
tf.random.set_seed(100)

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
	print(f'GPUs {gpus}')
	try: tf.config.experimental.set_memory_growth(gpus[0], True)
	except RuntimeError: pass

K.set_image_data_format("channels_first")


AAPL = pd.read_csv('output/AAPL.csv')
COST = pd.read_csv('output/COST.csv')
PEP = pd.read_csv('output/PEP.csv')
C = pd.read_csv('output/C.csv')
AAPL_norm = pd.read_csv('output/AAPL_norm.csv')
COST_norm = pd.read_csv('output/COST_norm.csv')
PEP_norm = pd.read_csv('output/PEP_norm.csv')
C_norm = pd.read_csv('output/C_norm.csv')


train_df = []
train_df_norm = []
train_df.append(AAPL.values)
train_df.append(COST.values)
train_df.append(PEP.values)
train_df.append(C.values)
train_df_norm.append(AAPL_norm.values)
train_df_norm.append(COST_norm.values)
train_df_norm.append(PEP_norm.values)
train_df_norm.append(C_norm.values)


xa = np.copy(np.moveaxis(np.array(train_df),0,-1))
x_norm = np.copy(np.moveaxis(np.array(train_df_norm),0,-1))
lookback_window_size = 50
print('shapedp',x_norm.shape, xa.shape)
train_df = xa[:-3000-lookback_window_size]
train_df_norm = x_norm[:-10000-lookback_window_size]
test_df = xa[-3000:] # 30 days
test_df_norm = x_norm[-3000:]


print('shape12345',train_df.shape)
model = 'EIIE'
train_env = CustomEnv(train_df, train_df_norm, lookback_window_size=lookback_window_size, model=model, stocks=['AAPL','COST','PEP','C'])
test_env = CustomEnv(test_df, test_df_norm,lookback_window_size=lookback_window_size, model=model, stocks=['AAPL','COST','PEP','C'])
agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, stocks=['AAPL','COST','PEP','C'], optimizer=Adam, batch_size = 32, model=model, shape = x_norm.shape)
train_agent(train_env, agent, visualize=False, train_episodes=50, training_batch_size=500)
test_agent(test_env, visualize=False, test_episodes=10, testing_batch_size=500)
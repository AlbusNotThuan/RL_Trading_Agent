import numpy as np
import tensorflow as tf
from collections import deque

@tf.function
def _train_step(agent, state, order):
    prediction = agent.act(state, order)
    return tf.squeeze(prediction)

def train_agent(env, agent, visualize=False, train_episodes=50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value, train_episodes) # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth

    for episode in range(train_episodes):
        #Reset the env
        state, order = env.reset(env_steps_size = training_batch_size)
        states, orders, rewards, predictions, dones, next_states, next_orders = [], [], [], [], [], [], []
        for t in range(training_batch_size):
            # Execute training step in graph mode
            prediction = _train_step(agent, 
                                   tf.convert_to_tensor(state, dtype=tf.float32),
                                   tf.convert_to_tensor(order, dtype=tf.float32))
            
            # Perform an action on env
            next_state, next_order, reward, done, prices = env.step(prediction.numpy())
            
            # Store as tensors
            states.append(tf.convert_to_tensor(state, dtype=tf.float32))
            orders.append(tf.convert_to_tensor(order, dtype=tf.float32))
            next_states.append(tf.convert_to_tensor(next_state, dtype=tf.float32))
            next_orders.append(tf.convert_to_tensor(next_order, dtype=tf.float32))
            
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
            order = next_order

        # Batch process tensors
        states = tf.stack(states)
        orders = tf.stack(orders)
        next_states = tf.stack(next_states)
        next_orders = tf.stack(next_orders)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        predictions = tf.stack(predictions)

        # Train the Model
        a_loss, c_loss = agent.replay(states, orders, rewards, predictions, dones, next_states, next_orders)
        total_average.append(env.net_worth)
        average = tf.reduce_mean(tf.cast(total_average, tf.float32))
        rewardFull = tf.reduce_mean(rewards)

        agent.writer.add_scalar('Data/average reward', rewardFull.numpy(), episode)
        agent.writer.add_scalar('Data/average net_worth', average.numpy(), episode)
        agent.writer.add_scalar('Data/average net_worth_percent', tf.round((average-1000)/10).numpy(), episode)

        ubah_value = tf.tensordot(env.quants_ubah, prices, axes=1)
        diff_value = env.net_worth - ubah_value

        print("net worth {} {:.2f} {:.2f} {:.2f} % UBAH {:.2f} diff {:.2f}".format(
            episode, 
            env.net_worth, 
            average.numpy(), 
            (average.numpy() - 1000) / 10, 
            ubah_value.numpy(), 
            diff_value.numpy()
        ))
        if episode > len(total_average):
          if best_average < average:
            best_average = average
            print("Saving model")
            agent.save(score="{:.2f}".format(best_average), args=[episode, average,  a_loss, c_loss])
        agent.save()

@tf.function
def test_step(agent, state, order):
    prediction = agent.act(state, order)
    return tf.squeeze(prediction)

@tf.function
def test_agent(env, agent, visualize=True, test_episodes=10, testing_batch_size=500):

    average_net_worth = tf.constant(0.0)
    average_UBAH = tf.constant(0.0)
    for episode in range(test_episodes):
        state, order = env.reset(env_steps_size=testing_batch_size)
        old = 0
        for t in range(testing_batch_size):
            prediction = test_step(agent, 
                                   tf.convert_to_tensor(state, dtype=tf.float32),
                                   tf.convert_to_tensor(order, dtype=tf.float32))
            old = env.net_worth
            state, order, reward, done, prices = env.step(prediction.numpy())

        average_net_worth += env.net_worth
        ubah_value = tf.tensordot(env.quants_ubah, prices, axes=1)
        diff_value = env.net_worth - ubah_value
        average_UBAH += ubah_value

        tf.print("net_worth:", env.net_worth, 
                "%", (env.net_worth - 1000) / 10, 
                "UBAH", ubah_value, 
                "diff", diff_value)

    tf.print("average:", average_net_worth / tf.cast(test_episodes, tf.float32),
             average_UBAH / tf.cast(test_episodes, tf.float32))
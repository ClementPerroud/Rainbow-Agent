import tensorflow as tf
import numpy as np
import gym
from agent import Rainbow

env = gym.make("LunarLander-v2")

def importance_weights_function(training_count, total_step):
    base = 0.4
    begin = 5_000 # Staying at base (0.4)
    linear = 30_000 # Going from 0.4 to 1
    if total_step < begin:
        return base 
    else:
        ratio = min(1, (total_step - begin)/linear )# From 0 to 1
        return base*(1 - ratio) + 1*ratio

def epsilon_function(training_count, total_step):
    begin = 5_000 # Staying at 1
    linear = 20_000 # Going from 1 to 0.01 linearly
    if total_step < begin:
        return 1
    if total_step < linear + begin:
        return 0.01 + 0.99*( begin + linear - total_step)/linear
    return 0.01 * 0.999**(total_step - begin - linear)  # Exponential decrease

rainbow = Rainbow(
    distributional= True, v_min= -200, v_max = 200, nb_atoms= 51, #Distributional
    adversarial= True, # Adversarial
    noisy= True, # Noisy
    windows = 5, # None recurrent
    nb_states = 8,
    nb_actions = 4,
    multi_steps= 3,
    gamma = 0.99,
    replay_capacity = 1E6,
    learning_rate = 5E-3,
    units = [32, 32],
    dropout = 0,
    l2_reg = None,
    tau = 3000,
    batch_size= 128,
    train_every = 4,
    prioritized_function = lambda td_errors : np.sqrt(td_errors), 
    importance_weights_function = importance_weights_function,
    epsilon_function = epsilon_function, # Ignore if noisy is True
)


while True:
    obs, info = env.reset()
    for _ in range(10000):
        rainbow.store_observation(obs)
        action = rainbow.e_greedy_pick_action_or_random(rainbow.get_inputs())        
        next_obs, reward, done, truncated, info = env.step(action)

        rainbow.store_experience(obs, action, reward, next_obs, done , truncated)
        rainbow.train()

        obs = next_obs
        if done or truncated:
            break

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import gym
from agent import Rainbow
import json
import argparse


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

def run(params_dir, episodes):
    env = gym.make("LunarLander-v2")

    with open(params_dir) as json_params:
        params = json.load(json_params)
    
    if params["prioritized_function"]: params["prioritized_function"] = lambda td_errors : np.sqrt(td_errors)
    else:  params["prioritized_function"] = lambda td_errors : 1

    if params["importance_weights_function"]: params["importance_weights_function"] = importance_weights_function
    else: params["importance_weights_function"] = lambda training_count, total_step : 1

    if params["epsilon_function"]: params["epsilon_function"] = epsilon_function
    params["epsilon_function"] = lambda training_count, total_step : 0

    rainbow = Rainbow(
        **params
    )

    for i in range(episodes):
        obs, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            rainbow.store_observation(obs)
            action = rainbow.e_greedy_pick_action_or_random(rainbow.get_inputs())        
            next_obs, reward, done, truncated, info = env.step(action)

            rainbow.store_experience(obs, action, reward, next_obs, done , truncated)
            rainbow.train()

            obs = next_obs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-params_dir", type=str, help = "Enter the path of your agent parameters JSON file")
    parser.add_argument("-episodes", type=int, help = "Enter number of episodes")
    parser.add_argument("-repeat", default = 1, type=int, help = "Enter number of episodes")
    args = parser.parse_args()

    for i in range(args.repeat):
        run(args.params_dir, args.episodes)
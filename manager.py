import tensorflow as tf
import numpy as np
import datetime
from .utils.memories import ReplayMemory, RNNReplayMemory, MultiStepsBuffer
from .utils.models import ModelBuilder, AdversarialModelAgregator
import os
import dill
import glob
import json
from .agent import Rainbow


class AgentManager:
    def __init__(self,
            make_agent,
            simultaneous_training_env = 1,
        ):
        self.simultaneous_training_env = simultaneous_training_env
        self.agents : list[Rainbow]= [make_agent() for _ in range(self.simultaneous_training_env)]

        self.losses = []
        self.episode_rewards = [[] for _ in range(self.simultaneous_training_env)]
        self.episode_steps = [0 for _ in range(self.simultaneous_training_env)]

        self.start_time = datetime.datetime.now()

    def new_episode(self, i_env):
        self.episode_steps[i_env] = 0
        self.episode_rewards[i_env] = []

    def store_replays(self, states, actions, rewards, next_states, dones, truncateds, states_trainable, next_states_trainable):
        for i_env in range(len(actions)):
            self.store_replay(
                state = states[i_env], action = actions[i_env], reward = rewards[i_env], next_state = next_states[i_env],
                done = dones[i_env], truncated = truncateds[i_env],
                state_trainable = states_trainable[i_env], next_state_trainable = next_states_trainable[i_env],
                i_env = i_env)
        
    def store_replay(self, state, action, reward, next_state, done, truncated, state_trainable, next_state_trainable, i_env = 0):
        self.agents[i_env].store_replay(
            state = state, action = action, reward = reward, next_state = next_state, done = done, truncated = truncated, 
            state_trainable = state_trainable, next_state_trainable = next_state_trainable
        )
                
        # Store history
        self.episode_rewards[i_env].append(reward)

        if done or truncated:
            self.agents[i_env].episode_count += 1
            self.log(i_env)
            self.new_episode(i_env)


    def train(self):
        for i in range(self.simultaneous_training_env): self.episode_steps[i] += 1
        loss_value = self.agents[0].train()
        for i in range(1, self.simultaneous_training_env): self.agents[i].steps += 1
        if loss_value is not None: self.losses.append(loss_value)

    
    def e_greedy_pick_action(self, state): return self.agents[0].e_greedy_pick_action(state = state)

    def e_greedy_pick_actions(self, states):
        epsilon = self.agents[0].get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.agents[0].nb_actions, size = self.simultaneous_training_env)
        return self.pick_actions(states).numpy()

    def pick_action(self, *args, **kwargs): return self.agents[0].pick_action(*args, **kwargs)

    def pick_actions(self, *args, **kwargs): return self.agents[0].pick_actions(*args, **kwargs)

    def log(self, i_env = 0):
        text_print =f"\
↳ Env {i_env} : {self.agents[i_env].episode_count:03} : {self.agents[i_env].steps: 8d}   |   {self.format_time(datetime.datetime.now() - self.start_time)}   |   Epsilon : {self.agents[0].get_current_epsilon()*100: 4.2f}%   |   Mean Loss (last 10k) : {np.mean(self.losses[-10_000:]):0.4E}   |   Tot. Rewards : {np.sum(self.episode_rewards[i_env]): 8.2f}   |   Rewards (/1000 steps) : {1000 * np.sum(self.episode_rewards[i_env]) / self.episode_steps[i_env]: 8.2f}   |   Length : {self.episode_steps[i_env]: 6.0f}"
        print(text_print)

    def format_time(self, t :datetime.timedelta):
        h = t.total_seconds() // (60*60)
        m  = (t.total_seconds() % (60*60)) // 60
        s = t.total_seconds() % (60)
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"

    def save(self, path, **kwargs):
        self.saved_path = path
        if not os.path.exists(path): os.makedirs(path)

        if self.agents[0].model is not None: self.agents[0].model.save(f"{path}/model.h5")
        if self.agents[0].target_model is not None: self.agents[0].target_model.save(f"{path}/target_model.h5")
        
        with open(f'{path}/agent_manager.pkl', 'wb') as file:
            dill.dump(self, file)
        for key, element in kwargs.items():
            if isinstance(element, dict):
                with open(f'{path}/{key}.json', 'w') as file:
                    dill.dump(element, file)
            else:
                with open(f'{path}/{key}.pkl', 'wb') as file:
                    dill.dump(element, file)


def load_agent(path):
    with open(f'{path}/agent_manager.pkl', 'rb') as file:
        unpickler = dill.Unpickler(file)
        agent_manager : AgentManager = unpickler.load()
    model = tf.keras.models.load_model(f'{path}/model.h5', compile=False, custom_objects = {"AdversarialModelAgregator" : AdversarialModelAgregator})
    target_model = tf.keras.models.load_model(f'{path}/target_model.h5', compile=False, custom_objects = {"AdversarialModelAgregator" : AdversarialModelAgregator})
    for i in range(len(agent_manager.agents)):
        agent_manager.agents[i].model = model
        agent_manager.agents[i].target_model = target_model

    other_elements = {}
    other_pathes = glob.glob(f'{path}/*pkl')
    other_pathes.extend(glob.glob(f'{path}/*json'))
    for element_path in other_pathes:
        name = os.path.split(element_path)[-1].replace(".pkl", "").replace(".json", "")
        if name != "agent":
            with open(element_path, 'rb') as file:
                if ".pkl" in element_path:other_elements[name] = dill.load(file)
                elif ".json" in element_path:other_elements[name] = json.load(file)
            
    return agent_manager, other_elements

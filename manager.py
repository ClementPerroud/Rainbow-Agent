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
            agent : Rainbow,
            simultaneous_training_env = 1,
        ):
        self.agent = agent
        self.simultaneous_training_env = simultaneous_training_env

        self.losses = []
        self.episode_rewards = [[] for _ in range(self.simultaneous_training_env)]
        self.episode_steps = [0 for _ in range(self.simultaneous_training_env)]

        self.start_time = datetime.datetime.now()

        
    def new_episode(self, i_env):
        self.episode_steps[i_env] = 0
        self.episode_rewards[i_env] = []
        
    def store_replay(self, state, action, reward, next_state, done, truncated, i_env = 0):
        self.agent.store_replay(state=state, action= action, reward= reward, next_state= next_state, done= done, truncated= truncated)
                    
        # Store history
        self.episode_rewards[i_env].append(reward)

        if done or truncated:
            self.log(i_env)
            self.new_episode(i_env)

    def train(self):
        for i_env in range(self.simultaneous_training_env): self.episode_steps[i_env] += 1
        loss_value = self.agent.train()
        if loss_value is not None: self.losses.append(loss_value)


    def store_replays(self, states, actions, rewards, next_states, dones, truncateds):
        for i_env in range(len(actions)):
            self.store_replay(states[i_env], actions[i_env], rewards[i_env], next_states[i_env], dones[i_env], truncateds[i_env], i_env = i_env)

    

    def e_greedy_pick_action(self, state): return self.agent.e_greedy_pick_action(state = state)

    def e_greedy_pick_actions(self, states):
        epsilon = self.agent.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.agent.nb_actions, size = self.simultaneous_training_env)
        return self.pick_actions(states).numpy()

    def pick_action(self, *args, **kwargs): return self.agent.pick_action(*args, **kwargs)

    def pick_actions(self, *args, **kwargs): return self.agent.pick_actions(*args, **kwargs)

    def log(self, i_env = 0):
        text_print =f"\
↳ Env {i_env} : {self.agent.episode_count:03} : {self.agent.steps: 8d}   |   {self.format_time(datetime.datetime.now() - self.start_time)}   |   Epsilon : {self.agent.get_current_epsilon()*100: 4.2f}%   |   Mean Loss (last 10k) : {np.mean(self.losses[-10_000:]):0.4E}   |   Tot. Rewards : {np.sum(self.episode_rewards[i_env]): 8.2f}   |   Rewards (/1000 steps) : {1000 * np.sum(self.episode_rewards[i_env]) / self.episode_steps[i_env]: 8.2f}   |   Length : {self.episode_steps[i_env]: 6.0f}"
        print(text_print)

    def format_time(self, t :datetime.timedelta):
        h = t.total_seconds() // (60*60)
        m  = (t.total_seconds() % (60*60)) // 60
        s = t.total_seconds() % (60)
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"

    def save(self, path, **kwargs):
        self.saved_path = path
        if not os.path.exists(path): os.makedirs(path)

        if self.agent.model is not None: self.agent.model.save(f"{path}/model.h5")
        if self.agent.target_model is not None: self.agent.target_model.save(f"{path}/target_model.h5")
        
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
    agent_manager.agent.model = tf.keras.models.load_model(f'{path}/model.h5', compile=False, custom_objects = {"AdversarialModelAgregator" : AdversarialModelAgregator})
    agent_manager.agent.target_model = tf.keras.models.load_model(f'{path}/target_model.h5', compile=False, custom_objects = {"AdversarialModelAgregator" : AdversarialModelAgregator})

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

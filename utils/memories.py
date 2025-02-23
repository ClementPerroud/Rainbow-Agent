import numpy as np
import tensorflow as tf
from .binary_heap import SumTree

class ReplayMemory():
    def __init__(self, capacity, nb_states, prioritized = True, alpha = 0.65, beta_function = None):
        self.capacity = int(capacity)
        self.nb_states= int(nb_states)
        self.i = 0
        self.states_memory = np.zeros(shape=(self.capacity, self.nb_states), dtype = np.float32)
        self.actions_memory = np.zeros(shape=(self.capacity,), dtype= np.int16)
        self.rewards_memory = np.zeros(shape=(self.capacity,), dtype= np.float32)
        self.states_prime_memory = np.zeros(shape=(self.capacity, self.nb_states), dtype= np.float32)
        self.done_memory = np.full(shape=(self.capacity,), fill_value=0, dtype= np.int16)

        self.prioritized = prioritized
        if self.prioritized:
            # self.priorities = np.full(shape=(self.capacity,), fill_value = 1E3, dtype= np.float32)
            self.priorities = SumTree(size= self.capacity)
            self.alpha = alpha
            self.beta_function = beta_function

    def store(self, s, a, r, s_p, done, s_trainable, s_p_trainable):
        if not s_trainable or not s_p_trainable: return 
        
        i = self.i % self.capacity
        self.states_memory[i] = s
        self.actions_memory[i] = a
        self.rewards_memory[i] = r
        self.states_prime_memory[i] = s_p
        self.done_memory[i] = 1 - int(done)
        if self.prioritized: self.priorities.add(i)
        self.i += 1
       
        
    def size(self):
        return min(self.i,self.capacity)
    
    def sample(self, batch_size, episode_count = None, steps = None):
        size = self.size()
        if self.prioritized:
            beta = self.beta_function(episode_count, steps)
            if beta >= 1: self.prioritized = False
            # probabilities = self.priorities[:size] /np.sum(self.priorities[:size])
            # batch = np.random.choice(size, size = batch_size,
            #     p = probabilities,
            #     replace=False
            # )
            batch = self.priorities.sample(batch_size)
            weights = (size * self.priorities[batch]/self.priorities.sum() ) ** (- beta)
            weights = weights / np.amax(weights)
        else:
            batch = np.random.choice(size, size = batch_size,
                replace=False
            )
            weights = 1
        return \
            batch,\
            self.states_memory[batch],\
            self.actions_memory[batch],\
            self.rewards_memory[batch],\
            self.states_prime_memory[batch],\
            self.done_memory[batch],\
            weights

    def update_priority(self, batch, td_errors):
        if self.prioritized: self.priorities[batch] = (np.abs(td_errors)+ 1)**self.alpha


class RNNReplayMemory(ReplayMemory):
    def __init__(self, window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.states_memory = np.zeros(shape=(self.capacity, self.window,self.nb_states), dtype = np.float32)
        self.states_prime_memory = np.zeros(shape=(self.capacity,self.window, self.nb_states), dtype= np.float32)

# class RNNBuffer:
#     def __init__(self, windows, nb_states):
#         self.windows = windows
#         self.nb_states = nb_states
#         self.reset()
#     def reset(self):
#         self.inputs = np.zeros(shape = (self.windows, self.nb_states))
#     def append(self, obs):
#         self.inputs[:-1] = self.inputs[1:]
#         self.inputs[-1] = obs
#         return self
#     def get(self):
#         return self.inputs

class MultiStepsBuffer:
    def __init__(self, multi_steps, gamma):
        self.multi_steps = multi_steps
        self.gamma_array = np.array([gamma ** i for i in range(self.multi_steps)])
        self.reset()
        
    def reset(self):
        self.states = [None]*self.multi_steps
        self.actions =  [None]*self.multi_steps
        self.rewards = np.zeros(shape=(self.multi_steps,))
        self.next_states = None
        self.dones = np.zeros(shape=(self.multi_steps,))
        self.states_trainable = np.zeros(shape=(self.multi_steps,))
        self.next_states_trainable = None


    def add(self, state, action, reward,  next_state, done, state_trainable, next_state_trainable):
        self.states.append(state)
        self.states = self.states[1:]

        self.actions.append(action)
        self.actions = self.actions[1:]

        self.rewards[0:-1] = self.rewards[1:]
        self.rewards[-1] = reward

        self.next_states = next_state

        self.dones[0:-1] = self.dones[1:]
        self.dones[-1] = done

        self.states_trainable[0:-1] = self.states_trainable[1:]
        self.states_trainable[-1] = state_trainable

        self.next_states_trainable = next_state_trainable

    def is_full(self):
        return self.states[0] is not None
    
    def get_multi_step_replay(self):
        return dict(
            s = self.states[0], 
            a = self.actions[0], 
            r = (self.rewards * self.gamma_array).sum(), 
            s_p = self.next_states, 
            done = (self.dones.sum() > 0), 
            s_trainable = self.states_trainable[0],
            s_p_trainable = self.next_states_trainable
        )


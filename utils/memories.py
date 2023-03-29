import numpy as np
import tensorflow as tf


class TransitionMemory():
    def __init__(self, capacity, nb_states, prioritized_function = lambda td_errors : 1):
        self.capacity = int(capacity)
        self.nb_states= int(nb_states)
        self.i = 0
        self.states_memory = np.zeros(shape=(self.capacity, self.nb_states), dtype = np.float32)
        self.actions_memory = np.zeros(shape=(self.capacity,), dtype= np.int16)
        self.rewards_memory = np.zeros(shape=(self.capacity,), dtype= np.float32)
        self.states_prime_memory = np.zeros(shape=(self.capacity, self.nb_states), dtype= np.float32)
        self.done_memory = np.full(shape=(self.capacity,), fill_value=0, dtype= np.int16)
        self.probabilities = np.full(shape=(self.capacity,), fill_value = 1E3, dtype= np.float32)
        self.prioritized_function = prioritized_function

    def store(self, s, a, r, s_p, done):
        i = self.i % self.capacity
        self.states_memory[i] = s
        self.actions_memory[i] = a
        self.rewards_memory[i] = r
        self.states_prime_memory[i] = s_p
        self.done_memory[i] = 1 - int(done)
        self.i += 1
        
    def size(self):
        return min(self.i,self.capacity)
    
    def sample(self, batch_size):
        size = self.size()
        batch = np.random.choice(size, size = batch_size, p = self.probabilities[:size]/np.sum(self.probabilities[:size]), replace=False)
        return \
            batch,\
            self.states_memory[batch],\
            self.actions_memory[batch],\
            self.rewards_memory[batch],\
            self.states_prime_memory[batch],\
            self.done_memory[batch]

    def update_probabilities(self, batch, td_errors):
        self.probabilities[batch] = self.prioritized_function(np.array(td_errors)) + 0.1

    def get_importance_weights(self, batch, beta = 0.4):
        size = self.size()
        weights = (1 / (size * self.probabilities[batch] / np.sum(self.probabilities[:size]))) ** beta
        return weights / np.amax(weights)

class LSTMTransitionMemory(TransitionMemory):
    def __init__(self, capacity, seq_len, nb_states, prioritized_function):
        super().__init__(capacity, nb_states, prioritized_function)
        self.seq_len = seq_len
        self.states_memory = np.zeros(shape=(self.capacity, self.seq_len,self.nb_states), dtype = np.float32)
        self.states_prime_memory = np.zeros(shape=(self.capacity,self.seq_len, self.nb_states), dtype= np.float32)

class ObservationBuffer:
    def __init__(self, windows, nb_states):
        self.windows = windows
        self.nb_states = nb_states
        self.reset()
    def reset(self):
        if self.windows is not None:
            self.inputs = np.zeros(shape = (self.windows, self.nb_states))
    def update(self, obs):
        if self.windows is not None:
            self.inputs[:-1] = self.inputs[1:]
            self.inputs[-1] = obs
            return self.inputs
        else: self.inputs = obs
    def get_inputs(self):
        return self.inputs

class BufferMultiSteps:
    def __init__(self, multi_steps, nb_states, gamma, windows):
        self.multi_steps = multi_steps
        self.nb_states = nb_states
        self.windows = windows if windows is not None else 1

        self.start = self.windows - 1
        self.dim = self.multi_steps + self.start
        self.gamma_array = np.array([gamma ** i for i in range(self.multi_steps)])
        self.reset()
        if self.windows == 1:
            self.get_multi_steps_memory = self._get_multi_steps_memory_no_windows
        else: self.get_multi_steps_memory = self._get_multi_steps_memory_windows
    def reset(self):
        self.state = np.zeros(shape = (self.dim, self.nb_states))
        self.action = np.zeros(shape = (self.dim,))
        self.reward = np.zeros(shape = (self.dim,))
        self.next_state = np.zeros(shape = (self.dim, self.nb_states))
        self.done = np.zeros(shape = (self.dim,))
        self.truncated = np.zeros(shape = (self.dim,))
        
        self.len = 0

    def _append_array(array, value):
        array[:-1] = array[1:]
        array[-1] = value
        return array

    def append(self, state, action, reward, next_state, done, truncated):
        if self.dim > 1:
            self.state[:-1] = self.state[1:]
            self.action[:-1] = self.action[1:]
            self.reward[:-1] = self.reward[1:]
            self.next_state[:-1] = self.next_state[1:]
            self.done[:-1] = self.done[1:]
            self.truncated[:-1] = self.truncated[1:]
        
        
        self.state[-1] = state
        self.action[-1] = action
        self.reward[-1] = reward
        self.next_state[-1] = next_state
        self.done[-1] = done
        self.truncated[-1] = truncated
        
        self.len += 1

    def _get_multi_steps_memory_windows(self):
        if self.len >= self.dim :
            return (
                self.state[0:self.windows], 
                self.action[self.windows-1],
                np.sum(self.reward[self.start:] * self.gamma_array),
                self.next_state[-self.windows:],
                self.done[-1]
            )
        return None
    def _get_multi_steps_memory_no_windows(self):
        if self.len >= self.dim :
            return (
                self.state[0], 
                self.action[0],
                np.sum(self.reward * self.gamma_array),
                self.next_state[-1],
                self.done[-1]
            )
        return None
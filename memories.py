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

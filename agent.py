import tensorflow as tf
import numpy as np
import datetime
from .utils.memories import ReplayMemory, RNNReplayMemory, MultiStepsBuffer
from .utils.models import ModelBuilder, AdversarialModelAgregator
import os
import dill
import glob
import json




class Rainbow:
    def __init__(self,
            model,
            target_model,
            replay_memory : ReplayMemory,
            nb_actions, 
            gamma, 
            batch_size,
            epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step), 
            # Model buildes
            window = 1, # 1 = Classic , 1> = RNN
            # Double DQN
            tau = 500, 
            # Multi Steps replay
            multi_steps = 1,
            # Distributional
            distributional = False, nb_atoms = 51, v_min= -200, v_max= 200,
            # Prioritized replay
            # Vectorized envs

            train_every = 1,
            name = "Rainbow",
        ):
        self.name = name
        self.nb_actions = nb_actions
        self.gamma =  tf.cast(gamma, dtype= tf.float32)
        self.epsilon_function = epsilon_function
        self.replay_memory = replay_memory 
        self.tau = tau
        self.batch_size = batch_size
        self.train_every = train_every
        self.multi_steps = multi_steps


        self.recurrent = window > 1
        self.window = window

        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max
    
        self.distributional = distributional

        # Memory
        self.multi_steps_buffer = MultiStepsBuffer(self.multi_steps, self.gamma)

        # Models
        self.model = model
        self.target_model = target_model
        target_model.set_weights(self.model.get_weights())



        # Initialize Tensorboard
        # self.log_dir = f"logs/{name}_{self.start_time.strftime('%Y_%m_%d-%H_%M_%S')}"
        # self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # History
        self.steps = 0
        self.episode_count = -1

        #INITIALIZE CORE FUNCTIONS
        # Distributional training
        if self.distributional:
            self.delta_z = (v_max - v_min)/(nb_atoms - 1)
            self.zs = tf.constant([v_min + i*self.delta_z for i in range(nb_atoms)], dtype= tf.float32)

    
    def store_replay(self, state, action, reward, next_state, done, truncated, state_trainable, next_state_trainable):
        # Case where no multi-steps:
        if self.multi_steps == 1:
            self.replay_memory.store(
                s = state, a = action, r = reward, s_p = next_state, d = done, state_trainable = state_trainable, next_state_trainable = next_state_trainable
            )
        else:
            self.multi_steps_buffer.add(
                state = state, action = action, reward = reward, next_state = next_state, done = done, 
                state_trainable = state_trainable, next_state_trainable = next_state_trainable
            )
            if self.multi_steps_buffer.is_full():
                self.replay_memory.store(
                    **self.multi_steps_buffer.get_multi_step_replay()
                )


    def train(self):
        self.steps += 1
        if self.replay_memory.size() < self.batch_size or self.get_current_epsilon() >= 1:
            return None
        
        if self.steps % self.tau == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        if self.steps % self.train_every == 0:
            batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = self.replay_memory.sample(
                self.batch_size,
                self.episode_count, 
                self.steps
            )

            loss_value, td_errors = self.train_step(states, actions, rewards, states_prime, dones, importance_weights)
            self.replay_memory.update_priority(batch_indexes, td_errors)

            return float(loss_value)

        return None
        # Tensorboard
        # with self.train_summary_writer.as_default():
        #     tf.summary.scalar('Step Training Loss', loss_value, step = self.total_stats['training_steps'])

    
    def get_current_epsilon(self, delta_episode = 0, delta_steps = 0):
        # if self.noisy: return 0
        return self.epsilon_function(self.episode_count + delta_episode, self.steps + delta_steps)

    def e_greedy_pick_action(self, state):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions)
        return self.pick_action(state)
    

    def format_time(self, t :datetime.timedelta):
        h = t.total_seconds() // (60*60)
        m  = (t.total_seconds() % (60*60)) // 60
        s = t.total_seconds() % (60)
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"
   

    def train_step(self, *args, **kwargs):
        if self.distributional: return self._distributional_train_step(*args, **kwargs)
        return self._classic_train_step(*args, **kwargs)
    
    def pick_action(self, *args, **kwargs):
        if self.distributional: return int(self._distributional_pick_action(*args, **kwargs))
        return int(self._classic_pick_action(*args, **kwargs))
    
    def pick_actions(self, *args, **kwargs):
        if self.distributional: return self._distributional_pick_actions(*args, **kwargs)
        return self._classic_pick_actions(*args, **kwargs)

    # Classic DQN Core Functions
    @tf.function
    def _classic_train_step(self, states, actions, rewards_n, states_prime_n, dones_n, weights):
        best_ap =tf.argmax(self.model(states_prime_n, training = False), axis = 1)
        max_q_sp_ap = tf.gather_nd(
            params = self.target_model(states_prime_n, training = False), 
            indices = best_ap[:, tf.newaxis],
            batch_dims=1)
        q_a_target = rewards_n + tf.cast(dones_n, dtype= tf.float32) *(self.gamma ** self.multi_steps) * max_q_sp_ap

        with tf.GradientTape() as tape:
            q_prediction = self.model(states, training = True)
            q_a_prediction = tf.gather_nd(
                params = q_prediction, 
                indices = tf.cast(actions[:, tf.newaxis], dtype= tf.int32),
                batch_dims=1)
            td_errors = tf.math.abs(q_a_target - q_a_prediction)
            loss_value = tf.math.reduce_mean(
                tf.square(td_errors)*tf.cast(weights, tf.float32)
            )
        self.model.optimizer.minimize(loss_value, self.model.trainable_weights, tape = tape)
        return loss_value, td_errors

    @tf.function
    def _classic_pick_actions(self, states):
        return tf.argmax(self.model(states, training = False), axis = 1)
    
    @tf.function
    def _classic_pick_action(self, state):
        return tf.argmax(self.model(state[tf.newaxis, ...], training = False), axis = 1)

    # Distributional Core Functions
    @tf.function
    def _distributional_pick_actions(self, states):
        return tf.argmax(self._distributional_predict_q_a(self.model,states, training = False), 1)

    @tf.function
    def _distributional_pick_action(self, state):
        return tf.argmax(self._distributional_predict_q_a(self.model,state[tf.newaxis, :], training = False), axis = 1)

    @tf.function
    def _distributional_predict_q_a(self, model, s, training = True):
        p_a = model(s, training = training)
        q_a = tf.reduce_sum(p_a * self.zs, axis = -1)
        return q_a

    @tf.function
    def _distributional_train_step(self, states, actions, rewards, states_prime, dones, weights):
        best_a__sp =tf.argmax(
            self._distributional_predict_q_a(self.model, states_prime, training = False),
            axis = 1)

        Tz = rewards[..., tf.newaxis] * tf.ones(shape=(self.batch_size, self.nb_atoms)) + tf.cast(dones[..., tf.newaxis], dtype = tf.float32) * (self.gamma ** self.multi_steps) * self.zs * tf.ones(shape=(self.batch_size,self.nb_atoms))
        Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)

        b_j = (Tz - self.v_min)/self.delta_z
        l = tf.math.floor(b_j)
        u = tf.math.ceil(b_j)

        p__max_ap_sp = tf.gather_nd(
            params = self.target_model(states_prime, training = False), 
            indices = best_a__sp[:, tf.newaxis],
            batch_dims=1)
        
        m_l = p__max_ap_sp * (u - b_j)
        m_u = p__max_ap_sp * (b_j - l)

        u = tf.cast(u, tf.int32)
        l = tf.cast(l, tf.int32)

        batch_indexes = (tf.range(self.batch_size)[..., tf.newaxis] * tf.ones(shape=(self.batch_size, self.nb_atoms), dtype= tf.int32))[..., tf.newaxis]

        l_ = tf.concat([batch_indexes, l[:, :,tf.newaxis]], axis= -1)
        u_ = tf.concat([batch_indexes, u[:, :,tf.newaxis]], axis= -1)

        m = tf.scatter_nd(l_, m_l[:], shape=(self.batch_size, self.nb_atoms,)) + tf.scatter_nd(u_, m_u[:], shape=(self.batch_size, self.nb_atoms,))

        with tf.GradientTape() as tape:
            p__s_a = tf.gather_nd(
                params = self.model(states, training = True), 
                indices = tf.cast(actions[:, tf.newaxis], dtype= tf.int32),
                batch_dims=1)
            td_errors = - tf.reduce_sum( m * tf.math.log(tf.clip_by_value(p__s_a , 1E-6, 1.0 - 1E-6)), axis = -1)
            td_errors_weighted = td_errors *tf.cast(weights, tf.float32)
            loss_value = tf.math.reduce_mean(td_errors_weighted)
        
        self.model.optimizer.minimize(loss_value, self.model.trainable_weights, tape = tape)
        return loss_value, td_errors

    def __getstate__(self):
        print("Saving agent ...")
        return_dict = self.__dict__.copy()
        return_dict.pop('model', None)
        return_dict.pop('target_model', None)
        return_dict.pop('replay_memory', None)
        return return_dict



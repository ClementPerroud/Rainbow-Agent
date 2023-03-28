import tensorflow as tf
import numpy as np
import datetime
from models import AdversarialModel
from memories import LSTMTransitionMemory

class Rainbow:
    def __init__(self,
            nb_states, nb_actions, gamma, 
            epsilon_function,
            replay_capacity, learning_rate,
            architecture, seq_len, dropouts, 
            l2_reg, tau, batch_size,
            prioritized_function = lambda td_errors : 1,
            importance_weights_function = lambda episode_count, nb_step : 0,
            log_dir = 'results', log_mean = 5, training = True, log_every = 1, train_every = 1, env = None,
            multi_steps=3,
            n_atoms = 51, v_min= -200, v_max= 200, atoms = None,
            
            *args, **kwargs):
        
        self.seq_len = seq_len
        self.dropouts = dropouts
        self.nb_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma =  tf.cast(gamma, dtype= tf.float32)
        self.epsilon_function = epsilon_function
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.tau = tau
        self.multi_steps = multi_steps
        self.batch_size = batch_size
        self.prioritized_function = prioritized_function
        self.importance_weights_function = importance_weights_function
        self.log_dir = log_dir
        self.log_mean = log_mean
        self.training = training
        self.log_every = log_every
        self.train_every = train_every
        self.env= env
        
        # Utils
        self.delta_z = (v_max - v_min)/(n_atoms - 1)
        self.zs = tf.constant([v_min + i*self.delta_z for i in range(n_atoms)], dtype= tf.float32)
        self.gamma_array = np.array([self.gamma ** i for i in range(self.multi_steps)])
        self.logger = None
        self.memory = LSTMTransitionMemory(self.replay_capacity, seq_len, self.nb_states, self.prioritized_function)
        
        # Create models
        self.input_shape = (self.seq_len, self.nb_states)
        self.model = self.build_model()
        self.model.build(self.input_shape)
        self.target_model = self.build_model(trainable = False)
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.build(self.input_shape)
        self.optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate, epsilon= 1.5E-4)

        # Initiate history
        self.total_stats = {
            "steps": 0,
            "training_steps": 0,
            "validation_steps": 0,
            "episode_count":0,
            "training_episode_count":0,
            "validation_episode_count":0,
            "losses" : []
        }
        
        self.episodes_history = {
            "mean_loss": [],
            "total_rewards": [],
            "steps": [],
            "is_eval": []
        }

        self.reset(update_history= False)     
        
    def reset(self, update_history = True):
        if update_history:
            self.episodes_history["total_rewards"].append(np.sum(self.episode_stats["rewards"]))
            self.episodes_history["steps"].append(self.episode_stats["steps"])
            self.episodes_history["mean_loss"].append(np.mean(self.episode_stats["losses"][1:]) if len(self.episode_stats["losses"]) > 1 else np.inf)
            self.episodes_history["is_eval"].append(self.episode_stats["is_eval"])
            
            self.total_stats["episode_count"]+= 1
            if self.episode_stats["is_eval"]:
                self.total_stats["validation_episode_count"] += 1
            else:self.total_stats["training_episode_count"] += 1
        
        self.episode_stats = {
            "steps":0,
            "rewards" : [],
            "losses": [],
            "is_eval": False
        }
        self.multi_steps_memory = {
            "s" : [],
            "a" : [],
            "r" : [],
            "s_prime" : [],
            "done" : [],
            "truncated": [],
        }

    def evaluation_mode(self):
        self.episode_stats["is_eval"]= True

    def build_model(self, trainable=True):
        inputs = tf.keras.layers.Input(shape=(self.seq_len, self.nb_states))
        shared_model = tf.keras.layers.LSTM(units=self.architecture[0],
                                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                return_sequences = not(1 == len(self.architecture)),
                                dropout = self.dropouts[0],
                                trainable=trainable)(inputs)
        for i, units in enumerate(self.architecture[1:], 1):
            shared_model = tf.keras.layers.LSTM(units=units,
                                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                return_sequences = not(i + 1 == len(self.architecture)),
                                dropout = self.dropouts[i],
                                trainable=trainable)(shared_model)
        value_stream = shared_model
        value_stream = tf.keras.layers.Dense(units = 64)(value_stream)
        value_stream = tf.keras.layers.Dense(units = self.nb_atoms)(value_stream)
        
        actions_stream = shared_model
        actions_stream = tf.keras.layers.Dense(units= 128)(actions_stream)
        actions_stream = tf.keras.layers.Dense(units= self.nb_atoms * self.nb_actions)(actions_stream)
        actions_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(actions_stream)

        model = AdversarialModel(inputs, value_stream, actions_stream)
        return model
    
    def get_current_epsilon(self, delta_episode = 0, delta_steps = 0):
        return self.epsilon_function(self.total_stats['training_episode_count'] + delta_episode, self.total_stats['training_steps'] + delta_steps)

    def e_greedy_pick_actions_or_random(self, states):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions, size =len(states))
        states = tf.convert_to_tensor(states, dtype= tf.float32)
        return self.pick_actions(states).numpy()

    def e_greedy_pick_action_or_random(self, state):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions)
        return int(self.pick_action(state).numpy())

    def format_time(self, t :datetime.timedelta):
        h = t.total_seconds() // (60*60)
        m  = (t.total_seconds() % (60*60)) // 60
        s = t.total_seconds() % (60)
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"
    @tf.function
    def pick_actions(self, states):
        return tf.argmax(self.predict_q_a(self.model,states, training = False), 1)

    @tf.function
    def pick_action(self, state):
        return tf.argmax(self.predict_q_a(self.model,state[tf.newaxis, :], training = False), axis = 1)

    @tf.function
    def predict_q_a(self, model, s, training = True):
        p_a = model(s, training = training)
        q_a = tf.reduce_sum(p_a * self.zs, axis = -1)
        return q_a

    def train(self):
        if self.memory.size() < self.batch_size or self.get_current_epsilon() >= 1:
            return
        if self.total_stats['training_steps'] % self.tau == 0:
            self.target_model.set_weights(self.model.get_weights())
        if self.total_stats['training_steps'] % self.train_every != 0:
            return
        batch_indexes, states, actions, rewards, states_prime, dones = self.memory.sample(self.batch_size)
        importance_weights = self.memory.get_importance_weights(batch_indexes, self.importance_weights_function(self.total_stats['training_episode_count'], self.total_stats['training_steps']))

        loss_value, td_errors = self.train_step(states, actions, rewards, states_prime, dones, importance_weights)
        self.memory.update_probabilities(batch_indexes, td_errors)


        self.episode_stats["losses"].append(float(loss_value))
        self.total_stats["losses"].append(float(loss_value))

    @tf.function
    def train_step(self, states, actions, rewards, states_prime, dones, weights):

        best_a__sp =tf.argmax(
            self.predict_q_a(self.model, states_prime, training = False),
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
            grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value, td_errors
       

    def store_experience(self, s, a, r, s_prime, done, truncated, evaluation = False):
        if not evaluation:
            self.multi_steps_memory["s"].append(s)
            self.multi_steps_memory["a"].append(a)
            self.multi_steps_memory["r"].append(r)
            self.multi_steps_memory["s_prime"].append(s_prime)
            self.multi_steps_memory["done"].append(done)
            self.multi_steps_memory["truncated"].append(truncated)

            if len(self.multi_steps_memory["s"]) < self.multi_steps:
                return
            elif len(self.multi_steps_memory["s"]) == self.multi_steps + 1:
                for key in self.multi_steps_memory.keys():
                    self.multi_steps_memory[key] = self.multi_steps_memory[key][1:]

            try:
                done_max = self.multi_steps_memory["done"].index(True) + 1
                done_n = True
            except ValueError:
                done_max = self.multi_steps
                done_n = False

            try:
                truncated_max = self.multi_steps_memory["truncated"].index(True) + 1
                truncated_n = True
            except ValueError:
                truncated_max = self.multi_steps
                done_n = False
            if truncated_max >= done_max:


                self.memory.store(
                    self.multi_steps_memory["s"][0],
                    self.multi_steps_memory["a"][0],
                    np.sum(np.array(self.multi_steps_memory["r"][:done_max]) * self.gamma_array[:done_max]),
                    self.multi_steps_memory["s_prime"][-1],
                    done_n
                )
        self.update_metrics(s, a, r, s_prime, done, truncated)

    def update_metrics(self, s, a, r, s_prime, done, truncated):
        self.episode_stats['rewards'].append(r)
        self.episode_stats['steps'] +=1
        self.total_stats['steps'] += 1
        if self.episode_stats['is_eval']:
            self.total_stats['validation_steps'] +=1
        else:
            self.total_stats['training_steps'] +=1
        if done or truncated:
            self.reset()
            self.log()

    def log(self):
        return 
import tensorflow as tf
import numpy as np
import datetime
from utils.memories import TransitionMemory, LSTMTransitionMemory, BufferMultiSteps, ObservationBuffer
from utils.models import ModelBuilder

class Rainbow:
    def __init__(self,
            nb_states, 
            nb_actions, 
            gamma, 
            epsilon_function,
            replay_capacity, 
            learning_rate, 
            units,
            dropout,
            l2_reg, 
            tau, 
            batch_size,
            multi_steps=3,
            windows= None,
            distributional = False, nb_atoms = 51, v_min= -200, v_max= 200,
            adversarial = False,
            noisy = False,
            prioritized_function = lambda td_errors : 1,
            importance_weights_function = lambda episode_count, nb_step : 0,
            train_every = 1,
            name = "Rainbow",
        ):
        self.name = name
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma =  tf.cast(gamma, dtype= tf.float32)
        self.epsilon_function = epsilon_function
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.units = units
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.tau = tau
        self.batch_size = batch_size
        self.prioritized_function = prioritized_function
        self.importance_weights_function = importance_weights_function
        self.train_every = train_every
        self.multi_steps = multi_steps
        self.windows = windows
        self.noisy = noisy

        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.recurrent = self.windows is not None
        self.distributional = distributional
        self.adversarial = adversarial


        # Memory tools
        self.multi_steps_memory = BufferMultiSteps(multi_steps = self.multi_steps, nb_states = self.nb_states, gamma =self.gamma, windows= self.windows)
        self.observartion_buffer = ObservationBuffer(self.windows, self.nb_states)

        # Recurrent
        if self.recurrent:
            self.memory = LSTMTransitionMemory(self.replay_capacity, windows, self.nb_states, self.prioritized_function)
        else:
            self.memory = TransitionMemory(replay_capacity, self.nb_states, prioritized_function= self.prioritized_function)

        
        
        # Distributional
        if self.distributional:
            self.delta_z = (v_max - v_min)/(nb_atoms - 1)
            self.zs = tf.constant([v_min + i*self.delta_z for i in range(nb_atoms)], dtype= tf.float32)
            self.train_step = tf.function(self._distributional_train_step)
            self.pick_action = tf.function(self._distributional_pick_action)
            self.pick_actions = tf.function(self._distributional_pick_actions)
        else:
            self.train_step = tf.function(self._train_step)
            self.pick_action = tf.function(self._pick_action)
            self.pick_actions = tf.function(self._pick_actions)
        # Models
        self.model_builder = ModelBuilder(
            units=self.units, dropout = self.dropout, nb_states= self.nb_states, nb_actions = self.nb_actions, l2_reg = self.l2_reg, recurrent= self.recurrent,
            windows = self.windows, distributional = self.distributional, nb_atoms= self.nb_atoms, adversarial= self.adversarial, noisy=self.noisy
            )
        self.input_shape = (self.nb_states,)
        self.model = self.model_builder.build_model()
        self.model.build(self.input_shape)
        self.target_model = self.model_builder.build_model(trainable= False)
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.build(self.input_shape)
        self.optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate, epsilon= 1.5E-4)

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

        self.reset(update= False)
        self.start_time = datetime.datetime.now()

        # Initialize Tensorboard
        self.log_dir = f"logs/{name}_{self.start_time.strftime('%Y_%m_%d-%H_%M_%S')}"
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)


    def evaluation_mode(self):
        self.episode_stats["is_eval"]= True
    
    def reset(self, update = True):
        if update:
            self.episodes_history["total_rewards"].append(np.sum(self.episode_stats["rewards"]))
            self.episodes_history["steps"].append(self.episode_stats["steps"])
            self.episodes_history["mean_loss"].append(np.mean(self.episode_stats["losses"][1:]) if len(self.episode_stats["losses"]) > 1 else np.inf)
            self.episodes_history["is_eval"].append(self.episode_stats["is_eval"])
            
            self.total_stats["episode_count"]+= 1
            if self.episode_stats["is_eval"]:
                self.total_stats["validation_episode_count"] += 1
            else:self.total_stats["training_episode_count"] += 1

            # Tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Episode Mean Loss', self.episodes_history["mean_loss"][-1], step=self.total_stats["training_episode_count"])
                tf.summary.scalar('Episode Total rewards', self.episodes_history["total_rewards"][-1], step=self.total_stats["training_episode_count"])    
                tf.summary.scalar('Episode Rewards / 1000 step', self.episodes_history['total_rewards'][-1]/self.episodes_history['steps'][-1]*1000, step=self.total_stats["training_episode_count"])    
                tf.summary.scalar('Episode Steps', self.episodes_history['steps'][-1], step=self.total_stats["training_episode_count"])    
        
        self.episode_stats = {
            "steps":0,
            "rewards" : [],
            "losses": [],
            "is_eval": False
        }
        self.multi_steps_memory.reset()
        self.observartion_buffer.reset()

    def store_experience(self, state, action, reward, next_state, done, truncated):
        self.multi_steps_memory.append(state, action, reward, next_state, done, truncated)
        data_to_store = self.multi_steps_memory.get_multi_steps_memory()

        if data_to_store is not None:
            self.memory.store(
                *data_to_store
            )
        self.update_metrics(state, action, reward, next_state, done, truncated)
        
    def update_metrics(self, state, action, reward, next_state, done, truncated):
        self.episode_stats['rewards'].append(reward)
        self.episode_stats['steps'] +=1
        self.total_stats['steps'] += 1
        if self.episode_stats['is_eval']:
            self.total_stats['validation_steps'] +=1
        else:
            self.total_stats['training_steps'] +=1
        if done or truncated:
            self.reset()
            self.log()
        
        # Tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Step Training Reward', reward, step = self.total_stats['training_steps'])
    
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

        loss_value = float(loss_value)
        self.episode_stats["losses"].append(loss_value)
        self.total_stats["losses"].append(loss_value)

        # Tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Step Training Loss', loss_value, step = self.total_stats['training_steps'])

    def log(self):
        if not self.episodes_history["is_eval"][-1]: # Training
            text_print =f"↳ {self.total_stats['episode_count']:03} | {self.format_time(datetime.datetime.now() - self.start_time)} | Epsilon : {self.get_current_epsilon(-1, 0)*100:0.2f}%  |  Mean Loss : {self.episodes_history['mean_loss'][-1]:0.4E}  |  Rewards (/1000 steps) : {self.episodes_history['total_rewards'][-1]/self.episodes_history['steps'][-1]*1000:0.2f}  |  Length : {self.episodes_history['steps'][-1] :0.0f}"
        print(text_print + '\x1b[0m')
    
    def get_current_epsilon(self, delta_episode = 0, delta_steps = 0):
        if self.noisy: return 0
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

    def store_observation(self, obs):
        self.observartion_buffer.update(obs)

    def get_inputs(self):
        return self.observartion_buffer.get_inputs()


    def format_time(self, t :datetime.timedelta):
        h = t.total_seconds() // (60*60)
        m  = (t.total_seconds() % (60*60)) // 60
        s = t.total_seconds() % (60)
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"


    @tf.function
    def _train_step(self, states, actions, rewards_n, states_prime_n, dones_n, weights):
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
            td_errors_weighted = td_errors*tf.cast(weights, tf.float32)
            loss_value = tf.math.reduce_mean(tf.square(td_errors_weighted))
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value, td_errors
    
    # Classic DQN Core Functions

    @tf.function
    def _pick_actions(self, states):
        return tf.argmax(self.model(states, training = False), axis = 1)
    
    @tf.function
    def _pick_action(self, state):
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
            grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value, td_errors


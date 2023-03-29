import tensorflow as tf
import tensorflow_addons as tfa

class DistributionalAdversarialModel(tf.keras.Model):
    def __init__(self, shared_model, value_stream, actions_stream, name = "AdversarialModel"):
        super().__init__(inputs=shared_model, outputs = {"value" : value_stream, "actions": actions_stream}, name = name)

    def call(self, *args, **kwargs):
        outputs = super().call(*args, **kwargs)
        outputs = outputs["value"][:, tf.newaxis, :] + outputs["actions"] - tf.math.reduce_mean(outputs["actions"], axis = -2, keepdims = True)
        return outputs
    
    def get_config(self):
        return super().get_config()

class ClassicAdversarialModel(tf.keras.Model):
    def __init__(self, shared_model, value_stream, actions_stream, name = "AdversarialModel"):
        super().__init__(inputs=shared_model, outputs = {"value" : value_stream, "actions": actions_stream}, name = name)

    def call(self, *args, **kwargs):
        outputs = super().call(*args, **kwargs)
        outputs = outputs["value"] + outputs["actions"] - tf.math.reduce_mean(outputs["actions"], axis = -1, keepdims=True)
        return outputs
    
    def get_config(self):
        return super().get_config()


class ModelBuilder():
    def __init__(self, units, dropout, nb_states, nb_actions, l2_reg, recurrent, windows, distributional, nb_atoms, adversarial, noisy):
        self.units = units
        self.dropout = dropout
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.l2_reg = l2_reg
        self.recurrent = recurrent
        self.windows = windows
        self.distributional = distributional
        self.nb_atoms = nb_atoms
        self.adversarial = adversarial
        self.noisy = noisy

        if self.noisy:
            self.dense = self._noisy_dense
        else:
            self.dense = self._dense

    
    def _dense(self, *args, **kwargs):
        return tf.keras.layers.Dense(*args, **kwargs)
    
    def _noisy_dense(self, *args, **kwargs):
        return tfa.layers.NoisyDense(*args, sigma= 0.5, **kwargs)

    def build_model(self, trainable = True):
        if self.recurrent: inputs = tf.keras.layers.Input(shape=(self.windows, self.nb_states))
        else : inputs = tf.keras.layers.Input(shape=(self.nb_states,))
        main_stream = inputs

        # Recurrent
        if self.recurrent:
            if self.noisy:
                main_stream = self._dense(units=self.units[0], activation = "tanh", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(main_stream)
            for i in range(1, len(self.units)):
                main_stream = tf.keras.layers.LSTM(units= self.units[i],
                                    return_sequences = not(i + 1 == len(self.units)),
                                    dropout = self.dropout,
                                    trainable=trainable)(main_stream)
        # Classic
        else:
            for i in range(len(self.units)):
                main_stream = self.dense(
                                    units= self.units[i],
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                    trainable=trainable)(main_stream)
                main_stream = tf.keras.layers.Dropout(self.dropout)(main_stream)
        
        # Distributional & Adversarial
        if self.distributional and self.adversarial:
            action_stream = main_stream
            action_stream = self.dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Dropout(self.dropout)(action_stream)
            action_stream = self.dense(units= self.nb_atoms * self.nb_actions, trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(action_stream)

            value_stream = main_stream
            value_stream = self.dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            value_stream = tf.keras.layers.Dropout(self.dropout)(value_stream)
            value_stream = self.dense(units = self.nb_atoms, trainable=trainable)(value_stream)

            adv_model = DistributionalAdversarialModel(inputs, value_stream, action_stream)
            output = adv_model(inputs)
            output = tf.keras.layers.Softmax(axis= -1)(output)
        
        # Only Distributional
        elif self.distributional and not self.adversarial:
            main_stream = self.dense(units= self.nb_atoms * self.nb_actions, trainable=trainable)(main_stream)
            main_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(main_stream)
            output = tf.keras.layers.Softmax(axis= -1)(main_stream)

        # Only Adversarial
        elif not self.distributional and self.adversarial:
            action_stream = main_stream
            action_stream = self.dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Dropout(self.dropout)(action_stream)
            action_stream = self.dense(units= self.nb_actions, trainable=trainable)(action_stream)

            value_stream = main_stream
            value_stream = self.dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            value_stream = tf.keras.layers.Dropout(self.dropout)(value_stream)
            value_stream = self.dense(units = 1, trainable=trainable)(value_stream)

            adv_model = ClassicAdversarialModel(inputs, value_stream, action_stream)
            output= adv_model(inputs)
        # Classic
        else:
            output = tf.keras.layers.Dense(units=self.nb_actions, trainable=trainable)(main_stream)
        

        model =  tf.keras.models.Model(inputs = inputs, outputs = output)
        return model
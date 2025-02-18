import tensorflow as tf
from .noisy_dense import NoisyDense

class AdversarialModelAgregator(tf.keras.Model):
    def call(self, outputs):
        # Distributional :
        # Value stream (batch, atoms)
        # Action stream (batch, actions, atoms)

        # Not Distributional (Classic)
        # Value stream (batch, )
        # Action stream (batch, actions)
        outputs = tf.expand_dims(outputs["value"], axis = 1) + outputs["actions"] - tf.math.reduce_mean(outputs["actions"], axis = 1, keepdims=True)
        return outputs



class ModelBuilder():
    def __init__(self, units, dropout, nb_states, nb_actions, l2_reg, window, distributional, nb_atoms, adversarial, noisy, learning_rate):
        self.units = units
        self.dropout = dropout
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.l2_reg = l2_reg
        self.recurrent = (window > 1)
        self.window = window
        self.distributional = distributional
        self.nb_atoms = nb_atoms
        self.adversarial = adversarial
        self.noisy = noisy
        self.learning_rate = learning_rate

    
    def dense(self, *args, **kwargs):
        if self.noisy: return NoisyDense(*args, sigma= 0.1, **kwargs)
        return tf.keras.layers.Dense(*args, **kwargs)
        

    def build_model(self, trainable = True):
        if self.recurrent: inputs = tf.keras.layers.Input(shape=(self.window, self.nb_states))
        else : inputs = tf.keras.layers.Input(shape=(self.nb_states,))
        main_stream = inputs

        # Recurrent
        if self.recurrent:
            #main_stream = self.dense(units=self.units[0], activation = "tanh", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(main_stream)
            for i in range(0, len(self.units)):
                main_stream = tf.keras.layers.LSTM(units= self.units[i],
                                    return_sequences = not(i + 1 == len(self.units)),
                                    dropout = self.dropout,
                                    trainable=trainable)(main_stream)
        # Classic
        else:
            for i in range(len(self.units)):
                main_stream = tf.keras.layers.Dense(
                                    units= self.units[i],
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                    trainable=trainable)(main_stream)
                if self.dropout > 0: main_stream = tf.keras.layers.Dropout(self.dropout)(main_stream)
        
        # Distributional & Adversarial
        if self.distributional and self.adversarial:
            action_stream = main_stream
            action_stream = tf.keras.layers.Dense(units = 512, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            if self.dropout > 0: action_stream = tf.keras.layers.Dropout(self.dropout)(action_stream)
            action_stream = self.dense(units= self.nb_atoms * self.nb_actions, trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(action_stream)

            value_stream = main_stream
            value_stream = tf.keras.layers.Dense(units = 512, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            if self.dropout > 0: value_stream = tf.keras.layers.Dropout(self.dropout)(value_stream)
            value_stream = self.dense(units = self.nb_atoms, trainable=trainable)(value_stream)

            output = AdversarialModelAgregator()({"value" :value_stream, "actions" : action_stream})
            output = tf.keras.layers.Softmax(axis= -1)(output)
        
        # Only Distributional
        elif self.distributional and not self.adversarial:
            main_stream = self.dense(units= self.nb_atoms * self.nb_actions, trainable=trainable)(main_stream)
            main_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(main_stream)
            output = tf.keras.layers.Softmax(axis= -1)(main_stream)

        # Only Adversarial
        elif not self.distributional and self.adversarial:
            action_stream = main_stream
            action_stream = tf.keras.layers.Dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Dropout(self.dropout)(action_stream)
            action_stream = self.dense(units= self.nb_actions, trainable=trainable)(action_stream)

            value_stream = main_stream
            value_stream = tf.keras.layers.Dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            value_stream = tf.keras.layers.Dropout(self.dropout)(value_stream)
            value_stream = self.dense(units = 1, trainable=trainable)(value_stream)

            output = AdversarialModelAgregator()({"value" :value_stream, "actions" : action_stream})[:, 0, :]
        # Classic
        else:
            output = tf.keras.layers.Dense(units=self.nb_actions, trainable=trainable)(main_stream)
        

        model =  tf.keras.models.Model(inputs = inputs, outputs = output)

        input_shape = (None, self.nb_states)
        model.build(input_shape)
        if trainable:
            model.compile(
                optimizer= tf.keras.optimizers.Adam(self.learning_rate, epsilon= 1.5E-4)
            )
        return model
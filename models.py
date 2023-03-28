import tensorflow as tf

class AdversarialModel(tf.keras.Model):
    def __init__(self, shared_model, value_stream, actions_stream, name = "AdversarialModel"):
        super().__init__(inputs=shared_model, outputs = {"value" : value_stream, "actions": actions_stream}, name = name)

    def call(self, *args, **kwargs):
        outputs = super().call(*args, **kwargs)

        outputs = outputs["value"][:, tf.newaxis, :] + outputs["actions"] - tf.reduce_mean(outputs["actions"], axis = -2, keepdims = True)
        
        outputs = tf.nn.softmax(outputs, axis = -1)
        return outputs
    
    def get_config(self):
        return super().get_config()
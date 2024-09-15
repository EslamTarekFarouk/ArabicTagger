import tensorflow as tf
from .CRF import CRF
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
class NER(tf.keras.Model):
    def __init__(self, units, k, n , m, **kwargs):
        """
        a NER (Named Entity Recognizer) model that uses a BI-LSTM + CRF layers to predict the tags of the words in a sentence
        units (int)---> number of hidden units in the LSTM
        k    (int)----> number of tags without START , END and padding tags
        n    (int)----> number of words in the sentence (maximum length)
        m    (int)----> number of features in the word embedding
        udt  (List[int])----> User Defined Tags (list of tags that the model should consider in the accuracy calculation)
        """
        self.units = units   
        self.k     = k     
        self.n     = n       
        self.m     = m
        self.udt   = kwargs.pop("udt", None)
        super(NER, self).__init__(**kwargs)
        # metrics to calculate the loss and accuracy
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
        self.udt_accuracy_tracker = tf.keras.metrics.Mean(name="udt_accuracy")
        self.lstm = Bidirectional(LSTM(self.units, return_sequences=True))
        self.crf  = CRF(k, n, self.units * 2)
    def call(self, data, training = False):
        # if training is True, the data is a tuple of X and Y
        # if training is False, the data is a tensor X which is used to predict the Y
        if training:
            X, Y = data
            return self.crf((self.lstm(X), Y))
        else:
            X = data 
            return self.crf(self.lstm(X), train = False)
        
    def train_step(self, data):
        x,y = data
        # if the shape of the input is (None, n+2, m) then we need to make it (1, n+2, m)
        if x[0].shape[0] == None:
            y   = tf.expand_dims(y[0], 0)
            x   = tf.expand_dims(x[0][0],0)
        else:
            x = x[0]
        # do forward pass
        with tf.GradientTape() as tape:
            y_hat, loss = self((x, y), training = True)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        accu = tf.reduce_mean(tf.cast(tf.equal(y, y_hat), tf.float32))
        # Calculate the accuracy of the UDT
        # if udt is None then use all the tags
        if self.udt:
            mask = tf.reduce_any(tf.equal(tf.expand_dims(y, axis=-1), self.udt), axis=-1)
            # Calculate temp using the mask
            temp = tf.cast(y == y_hat, dtype=tf.int32) * tf.cast(mask, dtype=tf.int32)
            temp = tf.reduce_sum(temp)
            # Calculate udt_accu
            udt_accu = temp / tf.reduce_sum(tf.cast(mask, dtype=tf.int32))
        else:
            udt_accu = accu
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "accuracy":
                metric.update_state(accu)
            elif metric.name == "udt_accuracy":
                metric.update_state(udt_accu)
            else:
                metric.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        x,y = data
        if x[0].shape == None:
            y   = tf.expand_dims(y[0], 0)
            x   = tf.expand_dims(x[0][0],0)
        else:
            x = x[0]
        # Compute predictions
        y_hat, loss = self((x, y), training = True)
        # Update the metrics.
        accu     = tf.reduce_mean(tf.cast(tf.equal(y, y_hat), tf.float32))
        if self.udt:
            mask = tf.reduce_any(tf.equal(tf.expand_dims(y, axis=-1), self.udt), axis=-1)
            # Calculate temp using the mask
            temp = tf.cast(y == y_hat, dtype=tf.int32) * tf.cast(mask, dtype=tf.int32)
            temp = tf.reduce_sum(temp)
            # Calculate udt_accu
            udt_accu = temp / tf.reduce_sum(tf.cast(mask, dtype=tf.int32))
        else:
            udt_accu  = accu
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "accuracy":
                metric.update_state(accu)
            elif metric.name == "udt_accuracy":
                metric.update_state(udt_accu)
            else:
                metric.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}
    
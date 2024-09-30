import tensorflow as tf
from .Multi_Head_Attention import Multi_Head_Attention

class Transformer_Encoder(tf.keras.layers.Layer):
    """
    Example
    --------
        # assume we a dataset of sentence each token either a device name or undefined token
        # assume that the dataset has 10,000 setences and the maximum length of the sentence is 10
        # our goal is to build a Encoder transformer model to tag the device names in the sentence
        # first use random inputs with embedding size of 3000
        inputs = tf.random.uniform((10000, 300, 10), -1, 1)
        tags   = tf.random.uniform((10000, 10), 0, 1, dtype = tf.int32)
        # build the model
        d = 300 
        n = 10
        h = 2
        m = 1
        input_  = Input((d,n))
        encoder = Transformer_Encoder(d, n, h, m)(input_)
        pooling = tf.keras.layers.GlobalAveragePooling1D()(encoder)
        output  = Dense(n, activation = "sigmoid")(pooling)
        model   = Model(input_, output)
        model.compile(optimizer="adam",
                      loss = "binary_crossentropy",
                      metrics=["binary_accuracy"])
        # now you can train the model
        #model.fit(X, Y, epochs=30, batch_size = 2**10, validation_split=0.2)
        
        # don't use this approach unless you have a large dataset
    """
    def __init__(self, d, n, h, m, **kwargs):
        """
        Transformer-Encoder Layer
        inputs:
            d (int)----> number of features in the word embedding
            n (int)----> number of words in the sentence (maximum length)
            h (int)----> number of heads
            m (int)----> number of recursive calls to the layer (Recursive Transformer)
        """
        self.d = d
        self.n = n
        self.h = h
        self.m = m
        # transformer encoder layers
        # in future updates the user will be able to customize these layers
        self.mhsa          = Multi_Head_Attention(self.d, self.n, self.h)
        self.add_normalize = tf.keras.layers.LayerNormalization(axis = -1, epsilon=0.01)
        self.NN_layer1     = tf.keras.layers.Dense(4*self.d, activation = "relu")
        self.NN_layer2     = tf.keras.layers.Dense(self.n, activation = "relu")
        self.dropout       = tf.keras.layers.Dropout(0.5)
        super(Transformer_Encoder, self).__init__(**kwargs)
    def build(self, input_shape):
        # intialize the parameters of each layer
        self.mhsa.build(input_shape)
        self.NN_layer1.build(input_shape)
        self.NN_layer2.build(self.NN_layer1.compute_output_shape(input_shape))
        super(Transformer_Encoder, self).build(input_shape)
    def call(self, X):
        for _ in range(self.m):
            mhsa          = self.mhsa(X)
            add_normalize = self.add_normalize(X + mhsa)
            NN_layer1     = self.NN_layer1(add_normalize)
            NN_layer2     = self.NN_layer2(NN_layer1) 
            output        = self.dropout(NN_layer2 + add_normalize)
            X             = output
        return output   
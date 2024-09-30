import tensorflow as tf
class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self, d, n, h, look_ahead_mask = False, **kwargs):
        """
        a multi head self attention layer
            d    (int)----> number of features in the word embedding
            n    (int)----> number of words in the sentence (maximum length)
            h    (int)----> number of heads
            look_ahead_mask (bool)----> if True the model will not be able to see the future words
            used in the decoder part of the transformer model
        for more information about the parameters check the paper "Attention is all you need"
        """
        self.d   = d
        self.n   = n
        self.h   = h
        self.k   = self.d / self.h
        if look_ahead_mask:
            self.look_ahead_mask = (tf.linalg.band_part(tf.ones((self.n, self.n)), -1, 0) - 1) * 30e30
        else:
            self.look_ahead_mask = tf.zeros((self.n, self.n))
        super(Multi_Head_Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        # intialize weights randomly using uniform distribution
        # Q, K, V, W are the weights of the model
        # all of the weights are expanded to enable batch maltiplication with the input matrix
        U_q = tf.random.uniform((self.h, self.d, self.d), minval = 0, maxval = 1)
        U_q = tf.expand_dims(U_q, 0)
        U_k = tf.random.uniform((self.h, self.d, self.d), minval = 0, maxval = 1)
        U_k = tf.expand_dims(U_k, 0)
        U_v = tf.random.uniform((self.h, self.d, self.d), minval = 0, maxval = 1)
        U_v = tf.expand_dims(U_v, 0)
        W   = tf.random.uniform((self.d, self.h * self.d), minval = 0, maxval = 1)
        W   = tf.expand_dims(W, 0)
        self.U_q = self.add_weight(shape = U_q.shape,\
                                initializer = tf.keras.initializers.Constant(U_q),\
                                trainable = True,\
                                dtype = tf.float32)
        self.U_k = self.add_weight(shape = U_k.shape,\
                                initializer = tf.keras.initializers.Constant(U_k),\
                                trainable = True,\
                                dtype = tf.float32)
        self.U_v = self.add_weight(shape = U_v.shape,\
                                initializer = tf.keras.initializers.Constant(U_v),\
                                trainable = True,\
                                dtype = tf.float32)
        self.W = self.add_weight(shape = W.shape,\
                                initializer = tf.keras.initializers.Constant(W),\
                                trainable = True,\
                                dtype = tf.float32)
    def call(self, x, encoder_inputs = False):
        """
        this function return multi head self attention tensor given the parameters and input matrix X
        inputs :
            X   (Tensor)-------------> input matrix of size [d,n]
            endocer_inputs (bool)----> if True the input is copied from the encoder and the decoder
        output :
            mhsa (Tensor)------------> matrix of size [d,n]
        """ 
        if encoder_inputs:
            # encoder output
            x1 = x[0]
            # decoder input
            x2 = x[1]
        else:
            x1 = x
            x2 = x
        batch_size =  tf.shape(x1)[0]
        # calculate the mask to prevent the paddings to slow down the model
        # get boolean vector to indicate position of padding  
        padding_mask = tf.math.not_equal(x1, 0)
        padding_mask = tf.reduce_all(padding_mask, axis = 1)
        # cast the vector from boolean to float32 then rshape it as column vector
        padding_mask = tf.cast(padding_mask, tf.float32)
        padding_mask = tf.reshape(padding_mask, (batch_size, 1, -1))
        # repeat that column n times 
        padding_mask = tf.concat([padding_mask]*self.n, axis = 1)
        padding_mask = (padding_mask - 1) * 30e30
        # add the look-ahead-mask
        mask = self.look_ahead_mask + padding_mask
        mask = tf.expand_dims(mask, 1)
        # caculate the multi-head attention
        x1   = tf.expand_dims(x1, 1)
        x2   = tf.expand_dims(x2, 1)
        Q    = tf.matmul(self.U_q, x2)  
        K    = tf.matmul(self.U_k, x1)
        V    = tf.matmul(self.U_v, x1)
        A    = tf.nn.softmax((tf.matmul(Q, K, transpose_a = True)  /tf.math.sqrt(self.k)) + mask ,\
                            axis = -1)
        Y    = V @ A
        mhsa = self.W @ tf.reshape(Y, [batch_size, self.h* self.d, self.n])
        return mhsa
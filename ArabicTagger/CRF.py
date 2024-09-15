import tensorflow as tf

class CRF(tf.keras.layers.Layer):
    def __init__(self, k, n, m,**kwargs):
        """
        a CRF (Conditional Random Field) layer
        for implementation details see: https://github.com/EslamTarekFarouk/ArabicTagger
        k    (int)----> number of tags without START , END and padding tags
        n    (int)----> number of words in the sentence (maximum length)
        m    (int)----> number of features in the word embedding
        """
        self.k = k
        self.m = m
        self.n = n
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        # intialize weights randomly using uniform distribution
        # W is represent each tag as a row and each feature as a column
        # W has a shape of (m, k+2)
        # E is the transition matrix between the tags (not T to prevent confusion with the time step)
        # E has a shape of (k+2, k+2)
        W = tf.random.uniform([self.m,self.k],0,1, dtype = tf.float64)
        W = tf.concat([tf.zeros([self.m,1], dtype = tf.float64), W, tf.zeros([self.m,1], dtype = tf.float64)], axis = 1)
        E = tf.random.uniform([self.k+1,self.k+1], 0, 1, dtype = tf.float64)
        E = tf.concat([E, tf.zeros([1,self.k+1], dtype = tf.float64)], axis = 0)
        E = tf.concat([tf.zeros([self.k+2,1], dtype = tf.float64), E], axis = 1)
        self.E = self.add_weight(shape = E.shape,\
                                initializer = tf.keras.initializers.Constant(E),\
                                trainable = True,\
                                dtype = tf.float64)
        self.W = self.add_weight(shape = W.shape,\
                                initializer = tf.keras.initializers.Constant(W),\
                                trainable = True,\
                                dtype = tf.float64)
    @tf.function
    def Veterbi(self, U):
        # for more information about the impementation of the Veterbi algorithm see: https://github.com/..../...
        labels  = tf.range(1, self.k+1, dtype=tf.int32)
        pi      = tf.zeros([labels.shape[0], self.n], dtype = tf.float64)
        # Base case 'initialization'
        pi = tf.tensor_scatter_nd_update(pi, [[i, 0] for i in range(self.k)], U[1, 1:-1])

        for t in range(2, self.n + 1):
            pi_t_minus_2 = tf.expand_dims(pi[:, t-2], axis=1)
            temp = pi_t_minus_2 + self.E[1:-1, 1:-1] + U[t, 1:-1]
            pi = tf.tensor_scatter_nd_update(pi, [[i, t-1] for i in range(self.k)], tf.reduce_max(temp, axis=0))

        # Backtracking
        best = tf.zeros((self.n + 2), dtype=tf.int32)
        best = tf.tensor_scatter_nd_update(best, [[self.n+1]], [labels[-1] + 1])
        best = tf.tensor_scatter_nd_update(best, [[self.n]], [labels[tf.argmax(pi[:, -1])]])
        for t in range(self.n - 1, 0, -1):
            last_best = best[t+1]
            temp_sum = pi[:, t-1] + self.E[1:-1, last_best] + U[t+1, last_best]
            best = tf.tensor_scatter_nd_update(best, [[t]], [labels[tf.argmax(temp_sum)]])
        return best
    @tf.function
    def Z_Forward(self, U):
        # for more information about the impementation of the Forward algorithm see: https://github.com/..../...
        # base case
        phi  = tf.reshape(tf.exp(U[1,1:-1]), [-1,1])
        for i in range(2, self.n+1):
            temp = tf.repeat(phi, phi.shape[0], axis = 1) * tf.exp(self.E[1:-1,1:-1] + U[i,1:-1])
            phi  = tf.reshape(tf.reduce_sum(temp, axis = 0), [-1,1])
        return tf.reduce_sum(phi)
    
    def call(self, data, train = True, return_loss = True):
        if train:
            # unpack the data
            inputs,Y = data
            # calculate unary potentials scores over the patch
            U     = tf.cast(inputs, tf.float64) @ self.W
            Y_hat = tf.map_fn(self.Veterbi, U, dtype = tf.int32, parallel_iterations = 11)
            # calculate the loss
            emissions   = tf.map_fn(fn =\
                          lambda y:tf.stack([y[1:-2], y[2:-1]], axis = 1), elems = Y)
            unaries     = tf.map_fn(fn =\
                          lambda y:tf.stack([tf.range(1,y.shape[0]-1), y[1:-1]], axis = 1),elems = Y)
            Z           = tf.map_fn(fn = self.Z_Forward, elems = U, parallel_iterations = 11)
            F           = tf.reduce_sum(tf.gather_nd(U, unaries, batch_dims = 1), axis = 1)
            F          += tf.reduce_sum(tf.gather_nd(self.E, emissions, batch_dims = 0), axis = 1)
            loss        = tf.reduce_mean(-1* (F - tf.math.log(Z)))
            if return_loss:
                return Y_hat,loss
            else:
                self.add_loss(loss)
                return Y_hat
        else:
            inputs = data
            U      = tf.cast(inputs, tf.float64) @ self.W
            Y_hat  = tf.map_fn(self.Veterbi, U, dtype = tf.int32, parallel_iterations = 11)
            return Y_hat
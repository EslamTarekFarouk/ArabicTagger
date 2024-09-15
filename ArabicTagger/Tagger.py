import numpy as np
import tensorflow as tf
from bpemb import BPEmb
import pickle
import os
from .NER import NER
class Tagger:
    """
        Tagging class is responsible for tagging the input text (Arabic)
        with the user defined tags + the tags of two pretrained models
        this are going to help he user to build a model knowing only the 
        tags (usually single tag).

        Example
        --------
        tagger = Tagger()
        tagger.intialize_models()
        inputs = [['السلام', 'عليكم', 'كم', 'سعر', 'الخلاط'],
                 ['ما', 'هي', 'مواصفات', 'البوتجاز', 'الي', 'في', 'الصورة']]
        tags =  [['DEVICE', 'O', 'O', 'O', 'O'],
                 ['O', 'O',' O', 'DEVICE', 'O', 'O', 'O']]
        # define udt
        user_defined_tags = ['DEVICE']
        # the output is a tuple of two tuples each tuple contains the input embeddings
        #  and the tags as integers 
        # the first tuple is the input embeddings and the tags of the first model
        # the second tuple is the input embeddings and the tags of the second model
        # you can use this data to train CRF model or NER(Bi-LSTM + CRF) model
        train_data1, train_data2 = tagger.get_data(inputs, 7, tags, user_defined_tags)
        # assume we have trained two model model1 & model2
        txt = "ما هي مواصفات الغسالة الي في الصورة"
        >>> tagger.tags1["DEVICE"]
        13
        >>> tagger.tags2["DEVICE"]
        8
        predictions = tagger.predict([model1, model2], txt.split(), [13, 8])
        # if the models are trained well the output should be
        #  array(['الغسالة'], dtype='<U7')
    """
    def __init__(self):
        pass
    def intialize_models(self):
        """
        intialize_models method is responsible for loading the pretrained models
        - BPEmb model which is responsible for embedding the input text 
        for more information about BPEmb model please visit https://github.com/bheinzerling/bpemb
        - CRF models & tags see the documentation of NER class at https://github.com/..../...
        """
        self.embedding =  BPEmb(dim = 300, lang = "arz")
        current_dir = os.path.dirname(__file__)
        self.model1   = self.load(os.path.join(current_dir, 'models', "CRF_model_1"))
        self.model2   = self.load(os.path.join(current_dir, 'models', "CRF_model_2"))
        self.tags1    = self.get_label_mapping(self.load_tags(os.path.join(current_dir, 'tags', 'tags1')))
        self.tags2    = self.get_label_mapping(self.load_tags(os.path.join(current_dir, 'tags', 'tags2')))
    def predict(self, models, words, tags_ids):
        """
        predict method is responsible for predicting the tags of the input list of words
        it makes sure to give you only the tags that are common between the models and which
        has been defined inside tags_ids list see examples for more information
        inputs:
            models (List[keras.Model])------> list of pretrained models (CRF or NER)
            words (List[str])---------------> list of words to be tagged
            tags_ids (List[int])------------> list of identical tags ids to be searched and filtered 
        outputs:
            prediction (numpy.ndarray)------> list of words with the predicted tags
        Example
        --------
        # assume we have defined Tagger object then prepared the data (see class example)
        # trained two models model1 and model2 
        >>> tagger.tags1["DEVICE"]
        13
        >>> tagger.tags2["DEVICE"]
        8
        # model 1 has assigned tag_id 8 to DEVICE and model 2 has assigned tag_id 13 to DEVICE
        # the model will only select token that has been assigned labels 8 and 13
        #  by model 1 and model 2 respectively
        txt = "ما هي مواصفات الغسالة الي في الصورة"
        predictions = tagger.predict([model1, model2], txt.split(), [13, 8])
        # if the model is trained well the output should be
        #  array(['الغسالة'], dtype='<U7')
        """
        target = 0 
        for model,tag_id in zip(models, tags_ids):
            X      = self.preprocess_input([words], model.n)
            Y_hat  = model.predict(X, verbose = False, batch_size = len(X))
            target = target + tf.cast(Y_hat == tag_id, tf.int32)
        target     = (target == len(models))[0,1:len(words)+1]
        prediction = np.array(words)[target.numpy()]
        return prediction
    def load_tags(self, path):
        # private method 
        # load tags from the file then return them
        # the tags stored as lists
        with open(f"{path}.pkl",'rb') as file:
            tags = pickle.load(file)
        return tags
    def load(self, path):
        # private method
        # load the model weights and parameters given the path
        # return the model
        # used to laod models 1 and 2
        with open(str(path)+".pkl", 'rb') as file:
            container = pickle.load(file)
        model = NER(container["units"], container["k"], container["n"], container["m"])
        model.lstm.build((None, container["n"], container["m"]))
        model.crf.build((None, container["n"], container["units"]*2))
        model.set_weights(container["model_weights"])
        print("model weights has been loaded!")
        return model
    def get_label_mapping(self, labels):
        # private method
        # map each lable to it's corresponding integer, starting from 1
        # save it to a dictionary then return it
        label_map   = dict() 
        for i,label in enumerate(labels):
            label_map[label] = i+1
        return label_map
    
    def vectorize_X(self, words):
        # private method
        # vectorize the input words using the embedding model BPEmb
        w_matrix    = []
        # start word are set of zeros
        w_matrix = [[0]*300]
        for word in words:
            embeds = self.embedding.embed(word)
            if len(embeds) > 1:
                embeds = np.average(embeds, axis = 0).reshape([1,embeds.shape[1]])
            w_matrix.append(embeds.tolist()[0])
        # end vector are set of 100s
        w_matrix.append([100]*300)
        return w_matrix
    
    def Normalize_length(self, x, padd, l):
        # private method
        # normalize the length of the input list x to be equal to l (maximum length)

        # if the length of the input list is less than maximum length append paddings
        # if the length of the input list is greater than maximum length truncate the list
        if len(x) < l:
            x.extend([padd]*(l - len(x)))
        elif len(x) > l:
            x = x[:l]
        return x
    def preprocess_input(self, words, n):
        # private method
        # preprocess the input words to be ready for the model

        # preprocessing steps are:
        # 1- normalize the length of the input list to be equal to n
        # 2- vectorize the input list using the embedding model
        # 3- return the vectorized input list as a tensor
        X = []
        for w in words:
            x = self.Normalize_length(w, "<PAD>", n)
            x = self.vectorize_X(x)
            X.append(x)
        X = tf.constant(X, dtype = tf.float64)
        return X
    def get_data(self, inputs, n, tag_lists, user_defiend_tags):
        """
        get_data method is responsible for preparing the data to be used in training
        the CRF model or NER model.
        inputs:
            inputs (List[List[str]])------> list of input text to be tagged "inputs can be iterable"
            n      (int)------------------> maximum length of the sentence 'n' must be less than or equal to 40
            tag_lists (List[List[str]])---> list of tags of the input text, the tags must be some unique tags 
            and the unknown tags are "O"  "tag_lists can be iterable"
            user_defined_tags (List[str])-> list of user defined tags whcih is unique tags in the tags lists except "O"
        outputs:
            train_data1 (Tuple)-----------> tuple of input embeddings Tensor and the tags of the first model
            train_data2 (Tuple)-----------> tuple of input embeddings Tensor and the tags of the second model
        Example
        --------
        tagger = Tagger()
        tagger.intialize_models()
        inputs = [['السلام', 'عليكم', 'كم', 'سعر', 'الخلاط'],
                 ['ما', 'هي', 'مواصفات', 'البوتجاز', 'الي', 'في', 'الصورة']]
        tags =  [['DEVICE', 'O', 'O', 'O', 'O'],
                 ['O', 'O',' O', 'DEVICE', 'O', 'O', 'O']]
        # define udt
        user_defined_tags = ['DEVICE']
        train_data1, train_data2 = tagger.get_data(inputs, 7, tags, user_defined_tags)
        >>> print(train_data1[0].shape, train_data1[1].shape)
        (2, 9, 300) (2, 9)

        """
        self.tags1 = self.get_label_mapping(list(self.tags1.keys())+user_defiend_tags)
        self.tags2 = self.get_label_mapping(list(self.tags2.keys())+user_defiend_tags)
        pos = []
        replacements = [[], []]
        for i, tag_list in enumerate(tag_lists):
            tag_list   = self.Normalize_length(tag_list, "O", self.model1.n)
            for j, tag in enumerate(tag_list):
                if tag != "O":
                    pos.append([i,j+1])
                    replacements[0].append(self.tags1[tag])
                    replacements[1].append(self.tags2[tag])
        X_1 = self.preprocess_input(inputs, self.model1.n)
        Y_1 = self.model1.predict(X_1, verbose = False, batch_size= len(X_1))
        Y_1 = tf.concat([tf.zeros([Y_1.shape[0], Y_1.shape[1]-1]),\
                        tf.ones([Y_1.shape[0], 1])],axis = 1) + Y_1
        Y_1  = tf.tensor_scatter_nd_update(Y_1, pos, replacements[0])
        Y_1  = tf.cast(Y_1, tf.int32)
        
        X_2  = self.preprocess_input(inputs, self.model2.n)
        Y_2  = self.model2.predict(X_2, verbose = False, batch_size= len(X_2))
        Y_2  = tf.concat([tf.zeros([Y_2.shape[0], Y_2.shape[1]-1]),\
                          tf.ones([Y_2.shape[0], 1])],axis = 1) + Y_2
        Y_2  = tf.tensor_scatter_nd_update(Y_2, pos, replacements[1])
        Y_2  = tf.cast(Y_2, tf.int32)
        # truncate the X's and Y's to the maximum length
        X_1  = tf.concat([X_1[:,:n+1,:], X_1[:,-1:,:]], axis = 1)
        X_2  = tf.concat([X_2[:,:n+1,:], X_2[:,-1:,:]], axis = 1)
        Y_1  = tf.concat([Y_1[:,:n+1], Y_1[:,-1:]], axis = 1)
        Y_2  = tf.concat([Y_2[:,:n+1], Y_2[:,-1:]], axis = 1)
        train_data1 = (X_1,Y_1)
        train_data2 = (X_2,Y_2)
        return train_data1, train_data2
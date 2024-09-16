import os
import pandas as pd
import pickle
import numpy as np
from nltk import everygrams
class TextMatcher:
    """
    TextMatcher is a module designed to address a common problem in Arabic NLP:
    Suppose you’re building a model to handle customer orders and process transactions automatically.
    You've implemented a Named Entity Recognition (NER) model to extract device names in Arabic,
    such as (غسالة, تلفاز, شاشة, مروحة).
    After identifying these devices, you need to match them with entries in your database to check stock availability.
    However, issues arise when there's a spelling variation.
    For instance, a customer might type "غساله," but in your database, the device is listed as "غسالة." 
    A standard search would fail to match these two, even though they represent the same item.
    This issue is known as orthographic or spelling variation and is a challenge for information retrieval in Arabic.
    Many approaches exist to address this, and in the ArabicTagger package, I approached the problem with a learnable weighted similarity.
    This method reduces the number of parameters and shortens training time while improving matching accuracy across spelling variations.
    for more information about the math behind the model
    see readme file in the github repository
    """
    def __init__(self, transformation_fun):
        """
        inputs:
        transformation_fun (fun)----> a function that is defined in [0,inf)
        this function is used to transform the position of the character in the word
        """
        self.transformation_fun = transformation_fun
        current_dir = os.path.dirname(__file__)
        directory   = os.path.join(current_dir, 'dict', "alphabets_indx.pkl")
        with open(directory,"rb")as file:
            self.alphabets_indx = pickle.load(file)
    def cosin_similiraty(self, x, y):
        # private function to calculate the cosine similarity between two vectors
        denominator = np.linalg.norm(x, axis = 0) * np.linalg.norm(y, axis = 0)
        denominator = denominator + 0.00001
        numerator   = np.sum(x * y, axis = 0) 
        return numerator/denominator
    def vectorize(self, words):
        # private function to vectorize the words
        # intialize a matrix as list of lists each list is a vector for a word
        # vector lenght is m = 1,260 
        matrix = []
        for word in words:
            vect = np.repeat(0.0001, len(self.alphabets_indx))
            # get every 1-gram and 2-gram of the word
            word_ng = everygrams(word, 1, 2)
            j = 0
            # the position of a single character is it's absolute position in the word
            # the position of a 2-gram is the average of the two characters positions
            for i,c in enumerate(word_ng):
                position = self.alphabets_indx.get(''.join(c))
                if (i+1)%2 == 0:
                    i = ((i-j) + (i-j+1))/2
                else:
                    i  = i - j
                    j += 1
                if position:
                    vect[position] += 1
                    vect[position] += self.transformation_fun(i)
            matrix.append(vect)
        X = np.array(matrix)
        return X
    
    def load_model(self, path = None):
        """
        load a pre-trained model or a model that was saved before
        input:
            path (str)-----> the path to the model must be the full path + name of file without any extension
            if the path is None a pre-trained model will be loaded 
        """
        if path:
            with open(str(path)+".pkl", 'rb') as file:
                self.W = pickle.load(file)
        else:
            current_dir = os.path.dirname(__file__)
            directory   = os.path.join(current_dir, 'models', "Matcher_weights1.pkl")
            with open(directory, 'rb') as file:
                container = pickle.load(file)
            self.W = container["W"]
            self.C = container["C"]
    def train(self):
        """
        the train method is used to train the model,must be called after fit method
        output:
            S_k (np.array)----> the similarity matrix for the positive class given the trained model
            S_j (np.array)----> the similarity matrix for the negative class given the trained model
            l_t (np.array)----> the labels for the positive class
            l_f (np.array)----> the labels for the negative class
        the outputs is used to calculate the optimal threshold C for the model and other metrics
        for more examples see kaggle notebook at github repository
        """
        w1_t, w1_f = self.train_data.query("l == True").w1, self.train_data.query("l == False").w1
        w2_t, w2_f = self.train_data.query("l == True").w2, self.train_data.query("l == False").w2
        l_t , l_f  = self.train_data.query("l == True").l, self.train_data.query("l == False").l
        I_k  = self.vectorize(w1_t)
        I_j  = self.vectorize(w1_f) 
        I_k_ = self.vectorize(w2_t)
        I_j_ = self.vectorize(w2_f)
        N_f  = 1 / (np.sum(self.cosin_similiraty(I_k , I_k_)) - \
                  np.sum(self.cosin_similiraty(I_j , I_j_)))
        W = N_f * (self.cosin_similiraty(I_k , I_k_) - self.cosin_similiraty(I_j , I_j_))
        W = W.reshape(-1,1)
        self.W = W
        S_k = (I_k * I_k_) @ (W * W)
        S_j = (I_j * I_j_) @ (W * W)
        return S_k,S_j,l_t,l_f
    
    def test(self):
        """
        the test method is used to test the model after training,
        must be called after train method
        output:
            S_k (np.array)----> the similarity matrix for the positive class given the trained model
            S_j (np.array)----> the similarity matrix for the negative class given the trained model
            l_t (np.array)----> the labels for the positive class
            l_f (np.array)----> the labels for the negative class
        the outputs is used to evaluate the model using the optimal threshold C on some metrics
        for more examples see kaggle notebook at github repository
        """
        w1_t, w1_f = self.test_data.query("l == True").w1, self.test_data.query("l == False").w1
        w2_t, w2_f = self.test_data.query("l == True").w2, self.test_data.query("l == False").w2
        l_t , l_f  = self.test_data.query("l == True").l, self.test_data.query("l == False").l
        I_k  = self.vectorize(w1_t)
        I_j  = self.vectorize(w1_f) 
        I_k_ = self.vectorize(w2_t)
        I_j_ = self.vectorize(w2_f)
        S_k  = (I_k * I_k_) @ (self.W * self.W)
        S_j  = (I_j * I_j_) @ (self.W * self.W)
        return S_k,S_j,l_t,l_f
    def predict(self, w1, w2, C):
        """
        predict method is used to predict the labels of the words
        give it two words that you want to compare and the optimal threshold C
        if the model is trained well it should predict the correct label
        1 means they are similar 0 means they are not
        inputs:
            w1 (List[str])----> the first word
            w2 (List[str])----> the second word
            C  (float)--------> the optimal threshold
        output:
            y_hat (np.array)--> the predicted labels
        """
        I_1   = self.vectorize(w1)
        I_2   = self.vectorize(w2) 
        S     = (I_1 * I_2) @ (self.W * self.W)
        y_hat = (S >= C) * 1
        return y_hat
    def fit(self, data, s):
        """
        fit method is used to split the data into training and testing sets 
        to prepare the data for training and testing the model
        inputs:
            data (pd.DataFrame)----> the data to split
            s    (float)-----------> the fraction of the data to be used for training s must be in (0,1)
        """
        data  = data.sample(frac = 1)
        s     = int(round(s * len(data)))
        self.train_data = data.iloc[:s,:]
        self.test_data  = data.iloc[s:,:]
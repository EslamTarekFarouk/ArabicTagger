import pickle
from .NER import NER
import math
def save(path, model):
    """
    Save the model parameters to a file
    inputs:
        path (str)-----> the path to save the model must be the full path + name of file without any extension
        model (NER)----> the model to save
    """
    container = {"k":model.k, "units":model.units,\
                 "m":model.m,"n":model.n,\
                 "model_weights":model.get_weights()}
    with open(str(path)+".pkl",'wb') as file:
        pickle.dump(container, file)
        print("model parameters has been saved !")


def load(path):
    """
    load the model parameters from a file
    input:
        path (str)-----> the path to the model must be the full path + name of file without any extension
    output:
        model (NER)----> the model with the loaded parameters
    """

    with open(str(path)+".pkl", 'rb') as file:
        container = pickle.load(file)
    model = NER(container["units"], container["k"], container["n"], container["m"])
    model.lstm.build((None, container["n"], container["m"]))
    model.crf.build((None, container["n"], container["units"]*2))
    model.set_weights(container["model_weights"])
    print("model weights has been loaded!")
    return model

def golden_section(fun, inte, ite = 10):
    """
    this function is used to find the minimum of a function in a given interval
    using the golden section search algorithm.
    inputs:
        fun (fun): the function to minimize
        inte (tuple): the interval to search in
        ite (int): the number of iterations
    output:
        res (float): the minimum of the function
    """
    lower,upper = inte
    r   = (math.sqrt(5) - 1)/2
    d   = r * (upper - lower)
    x1  = lower + d
    x2  = upper - d 
    counter = 1
    while(counter != ite):
        if fun(x1) > fun(x2):
            upper = x1
            x1    = x2
            d     = r * (upper - lower)
            x2    = upper - d
        else :
            lower = x2
            x2    = x1
            d     = r * (upper - lower)
            x1    = lower + d
        counter += 1
    res = (x1+x2)/2
    return res

def positional_embeddings(d, n, c = 10000):
    """
    get the positional embeddings of embeddings of size [d,n]
    inputs :
        d (int) -------------------> is the diemnsion of the embeddings
        n (int) -------------------> is the number of tokens in the sentence
        c (int) -------------------> hyperparameter by default 10000
    output :
        tokens (List[list[]])------> positional embeddings for the tokens as list of lists
    """
    tokens = []
    for position in range(n):
        tokens.append([])
        for i in range(d):
            # if i is even calculate even_pe
            if i%2 == 0:
                even_d    = math.pow(c, i/d)
                even_pe = math.sin(position/ even_d)
                tokens[position].append(even_pe)
            # if i is odd calculate odd_pe    
            else:
                odd_d    = math.pow(c, i/d)
                odd_pe = math.cos(position/ odd_d)
                tokens[position].append(odd_pe)            
    return tokens
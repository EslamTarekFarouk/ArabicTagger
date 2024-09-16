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
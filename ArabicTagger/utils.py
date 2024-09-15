import pickle
from .NER import NER

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
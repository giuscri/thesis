import pickle

def loadpickle(filename):
    """Load a Python object serialized using pickle at `filename'."""
    with open(filename, 'rb') as f:
        try: return pickle.load(f)
        except pickle.UnpicklingError: return None

import pickle

class Tools:
    @staticmethod
    def save_to_file(path, data):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(path):
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)
import os
import pickle

from src.data_transform import get_data


class GraphData:
    '''
    A class to handle graph data loading and caching create graph data with different split_ratio and  perturbation rate.
    '''
    def __init__(self, root='data', save_dir='cache'):
        self.root = root
        self.save_dir = os.path.join(root, save_dir)

    def load_graph(self, name, split_ratio, perturbation_rate):
        """
        Load the graph data from a pickle file.
        """
        file_path = os.path.join(self.save_dir, f"{name}_{split_ratio}_{perturbation_rate}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            data = get_data(self.root, dataset_name=name, false_pos_edge_ratio=perturbation_rate)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return data

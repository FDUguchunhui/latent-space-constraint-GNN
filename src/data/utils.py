import os
import pickle

from src.data.data_transform import get_data


class GraphData:
    '''
    A class to handle graph data loading and caching create graph data with different target_ratio and  perturbation rate.
    '''
    def __init__(self, root='data', save_dir='cache'):
        self.root = root
        self.save_dir = os.path.join(root, save_dir)

    def load_graph(self, name, target_ratio, perturb_rate, perturb_type='Random'):
        """
        Load the graph data from a pickle file.
        """
        file_path = os.path.join(self.save_dir, f"{name}_{target_ratio}_{perturb_rate}_{perturb_type}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            data = get_data(self.root, dataset_name=name, target_ratio=target_ratio, perturb_rate=perturb_rate, perb_type=perturb_type)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return data

import os
import numpy as np
import torch
import torch.nn as nn

from vocml_featurizers.paths import OBP_CACHE
from vocml_featurizers.featurizer_manager import FeaturizerBaseClass

class OneHotFeaturizer(nn.Module, FeaturizerBaseClass):
    def __init__(self, max_length=370, cache_file:str=os.path.join(OBP_CACHE, 'OneHotFeaturizer-mapping.pkl')):
        super(OneHotFeaturizer, self).__init__()

        # Fixed amino acid alphabet mapping
        self.mapping = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20, 'UNK': 20, '-': 21  # UNK & Gap
        }

        self.max_length = max_length  # Can be set manually or computed in _fit()
        self.cache_file = cache_file
        self.name = 'OneHotFeaturizer'
        self.cache = self._load_cache(cache_file)
        self.data_column = 1

    def _featurization_logic(self, to_featurize:list[str]) -> np.ndarray:
        '''
        Featurizes a protein sequence into a one-hot encoded representation.

        Args:
            data_loader: torch.utils.data.DataLoader - The data to be featurized
            train: bool - Whether to fit the featurizer before encoding

        Returns:
            np.ndarray: One-hot encoded representation of input sequences.
        '''

        all_encoded_data = []

        for token in to_featurize:
            one_hot_matrix = np.zeros((self.max_length, len(self.mapping)), dtype=np.float32)

            for i, char in enumerate(token[:self.max_length]):
                index = self.mapping.get(char, self.mapping["UNK"])  # Use "UNK" if unknown
                one_hot_matrix[i, index] = 1  # Set one-hot encoding
            tensor = torch.from_numpy(one_hot_matrix.flatten())
            self._store_in_cache(token, tensor)
            all_encoded_data.append(tensor)

        return torch.tensor(np.array(all_encoded_data))
    

class CategoricalFeaturizer(nn.Module, FeaturizerBaseClass):
    def __init__(self, max_length=370, cache_file:str=os.path.join(OBP_CACHE, 'CategoricalFeaturizer-mapping.pkl')):
        super(CategoricalFeaturizer, self).__init__()

        # Fixed amino acid alphabet mapping
        self.mapping = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20, 'UNK': 20, '-': 21  # UNK & Gap
        }

        self.max_length = max_length  # Can be set manually or computed in _fit()
        self.cache_file = cache_file
        self.name = 'CategoricalFeaturizer'
        self.cache = self._load_cache(cache_file)
        self.data_column = 1

    def _featurization_logic(self, to_featurize:list[str]) -> np.ndarray:
        '''
        Featurizes a protein sequence into a one-hot encoded representation.

        Args:
            data_loader: torch.utils.data.DataLoader - The data to be featurized
            train: bool - Whether to fit the featurizer before encoding

        Returns:
            np.ndarray: One-hot encoded representation of input sequences.
        '''

        all_encoded_data = []

        for token in to_featurize:
            one_hot_matrix = np.zeros(self.max_length, dtype=np.float32)

            for i, char in enumerate(token[:self.max_length]):
                index = self.mapping.get(char, self.mapping["UNK"])  # Use "UNK" if unknown
                one_hot_matrix[i] = index  # Set one-hot encoding
            tensor = torch.from_numpy(one_hot_matrix)
            self._store_in_cache(token, tensor)
            all_encoded_data.append(tensor)

        return torch.tensor(np.array(all_encoded_data))
    

if __name__ == "__main__":
    smiles = np.array(["ABC", "BBBB", "CCC"])

    cbfeats = OneHotFeaturizer()
    feats = cbfeats.featurize(smiles)

    print(feats, len(feats))
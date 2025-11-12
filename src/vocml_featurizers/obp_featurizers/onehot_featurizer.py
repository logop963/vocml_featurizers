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
            '-': 0,  # Gap
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
            'X': 21, 'UNK': 21  # UNK
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
            '-': 0,  # Gap
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
            'X': 21, 'UNK': 21  # UNK
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
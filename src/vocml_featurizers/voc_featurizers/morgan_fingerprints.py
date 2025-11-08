import numpy as np
import rdkit
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import os

from vocml_featurizers.featurizer_manager import FeaturizerBaseClass
from vocml_featurizers.paths import VOC_CACHE
class MorganFingerprintFeaturizer(nn.Module, FeaturizerBaseClass):
    def __init__(self, radius: int = 2, n_bits: int = 1024, cache_file:str=os.path.join(VOC_CACHE, 'MorganFingerrint-mapping.pkl')):
        self.radius  = radius
        self.n_bits = n_bits
        self.cache_file = cache_file
        self.name = 'CategoricalFeaturizer'
        self.cache = self._load_cache(cache_file)
        self.gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        
    def _featurization_logic(self, smiles_array: np.ndarray) -> np.ndarray:
        """Generate Morgan fingerprints from a numpy array of SMILES strings.

        Args:
            smiles_array (np.ndarray): Array of SMILES strings.
            radius (int): Radius of the Morgan fingerprint. Default is 2.
            n_bits (int): Size of the fingerprint vector. Default is 2048.

        Returns:
            np.ndarray: 2D array of Morgan fingerprints (n_samples x n_bits).
        """
        fps = []
        for smi in smiles_array:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                fp = np.zeros(self.n_bits, dtype=int)
            else:
                bitvect = self.gen.GetFingerprint(mol)
                arr = np.zeros((self.n_bits,), dtype=int)
                ConvertToNumpyArray(bitvect, arr)
                fp = arr
                self._store_in_cache(smi, torch.tensor(fp))
            fps.append(fp)

        return torch.tensor(np.array(fps))


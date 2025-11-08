from vocml_featurizers.obp_featurizers.protBERT_featurizer import ProtBERTFeaturizer
from vocml_featurizers.voc_featurizers.chemBERTa_featurizer import ChemBERTaFeaturizer
from vocml_featurizers.obp_featurizers.onehot_featurizer import CategoricalFeaturizer
from vocml_featurizers.voc_featurizers.morgan_fingerprints import MorganFingerprintFeaturizer

import torch
import torch.nn.functional as F

class TransformerFeaturizer:
    def __init__(self, pooling):
        self.voc_featurizer = ChemBERTaFeaturizer(pooling=pooling)
        self.obp_featurizer = ProtBERTFeaturizer(pooling=pooling)

    def __call__(self, batch):
        vocs = F.normalize(self.voc_featurizer.featurize(batch["smiles"]), p=2, dim=1)
        ors = F.normalize(self.obp_featurizer.featurize(batch["mutated_sequence"]), p=2, dim=1)
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["responsive"]}


class FPFeaturizer:
    def __init__(self, radius, n_bits):
        self.voc_featurizer = MorganFingerprintFeaturizer(radius=radius, n_bits=n_bits)
        self.obp_featurizer = CategoricalFeaturizer()

    def __call__(self, batch):
        vocs = self.voc_featurizer.featurize(batch["smiles"])
        ors = self.obp_featurizer.featurize(batch["mutated_sequence"])
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["responsive"]}
    

class MorganProtFeaturizer:
    def __init__(self, radius, n_bits):
        self.voc_featurizer = MorganFingerprintFeaturizer(radius=radius, n_bits=n_bits)
        self.obp_featurizer = ProtBERTFeaturizer()

    def __call__(self, batch):
        vocs = self.voc_featurizer.featurize(batch["smiles"])
        ors = F.normalize(self.obp_featurizer.featurize(batch["mutated_sequence"]), p=2, dim=1)
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["responsive"]}
    
class ChembertaCatFeaturizer:
    def __init__(self, pooling="mean"):
        self.voc_featurizer = ChemBERTaFeaturizer(pooling=pooling)
        self.obp_featurizer = CategoricalFeaturizer()

    def __call__(self, batch):
        vocs = F.normalize(self.voc_featurizer.featurize(batch["voc_smiles"]), p=2, dim=1)
        ors = self.obp_featurizer.featurize(batch["mutated_sequence"])
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["responsive"]}
    

class OBPsTransformerFeaturizer:
    def __init__(self, pooling):
        self.voc_featurizer = ChemBERTaFeaturizer(pooling=pooling)
        self.obp_featurizer = ProtBERTFeaturizer(pooling=pooling)

    def __call__(self, batch):
        vocs = F.normalize(self.voc_featurizer.featurize(batch["voc_smiles"]), p=2, dim=1)
        ors = F.normalize(self.obp_featurizer.featurize(batch["obp_seq"]), p=2, dim=1)
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["bind"]}
    
class OBPsFPFeaturizer:
    def __init__(self, radius, n_bits):
        self.voc_featurizer = MorganFingerprintFeaturizer(radius=radius, n_bits=n_bits)
        self.obp_featurizer = CategoricalFeaturizer()

    def __call__(self, batch):
        vocs = self.voc_featurizer.featurize(batch["voc_smiles"])
        ors = self.obp_featurizer.featurize(batch["obp_seq"])
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["bind"]}
    

class OBPsMorganProtFeaturizer:
    def __init__(self, radius, n_bits):
        self.voc_featurizer = MorganFingerprintFeaturizer(radius=radius, n_bits=n_bits)
        self.obp_featurizer = ProtBERTFeaturizer()

    def __call__(self, batch):
        vocs = self.voc_featurizer.featurize(batch["voc_smiles"])
        ors = F.normalize(self.obp_featurizer.featurize(batch["obp_seq"]), p=2, dim=1)
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["bind"]}
    
class OBPsChembertaCatFeaturizer:
    def __init__(self, pooling="mean"):
        self.voc_featurizer = ChemBERTaFeaturizer(pooling=pooling)
        self.obp_featurizer = CategoricalFeaturizer()

    def __call__(self, batch):
        vocs = F.normalize(self.voc_featurizer.featurize(batch["voc_smiles"]), p=2, dim=1)
        ors = self.obp_featurizer.featurize(batch["obp_seq"])
        inputs = torch.cat([vocs, ors], dim=1)
        return {"inputs": inputs, "targets": batch["bind"]}
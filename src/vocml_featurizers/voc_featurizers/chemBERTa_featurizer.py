import os
import torch

import torch.nn as nn
import numpy as np

from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, AutoTokenizer

from vocml_featurizers.featurizer_manager import FeaturizerBaseClass
from vocml_featurizers.paths import VOC_CACHE

class ChemBERTaFeaturizer(nn.Module, FeaturizerBaseClass):
    def __init__(self, pooling="mean", feats_cache_file:str=os.path.join(VOC_CACHE, "ChemBERTa-mapping.pkl")):
        super(ChemBERTaFeaturizer, self).__init__()
        self.pooling= pooling
        # Initialize tokenizer & model and give cache dir
        cache_dir = os.path.join(VOC_CACHE, 'chemBERTaFeaturizer')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        self.tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", do_lower_case=False, cache_dir=cache_dir)
        print("[ChemBERTa] Loading model...")
        self.model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir=cache_dir)
        print("[ChemBERTa] Model loaded successfully.")

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Identifiers
        self.name = "ChemBERTa"
        self.data_column = 0
        # Caching setup
        self.cache_file = os.path.join(VOC_CACHE, f"ChemBERTa-mapping-{pooling}.pkl")
        self.cache = self._load_cache(self.cache_file)
        print(f"[ChemBERTa] Using cache with {len(self.cache)} entries")



    def _tokenize(self, to_tokenize):
        """Tokenizes sequences, using the instanced tokenizer"""
        tokenized_outputs = {}

        new_tokens = self.tokenizer(
            to_tokenize, padding=True, truncation=True, max_length=176, return_tensors="pt"
        )
        for i, seq in enumerate(to_tokenize):
            tokenized_outputs[seq] = {key: val[i] for key, val in new_tokens.items()}

        # Convert to batch format
        batch = {key: torch.stack([tokenized_outputs[seq][key] for seq in to_tokenize]) for key in tokenized_outputs[to_tokenize[0]]}
        return batch
    
    def _masked_mean_pooling(self, embeddings, mask):
        mask  = mask.unsqueeze(-1).type_as(embeddings)
        return (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    
    def _featurization_logic(self, to_featurize:np.ndarray[str]) -> torch.Tensor:
        with torch.no_grad():
            tokenized_inputs = self._tokenize(to_featurize)  # Tokenize with cache

            # Move to correct device
            tokenized_inputs = {key: val.to(self.device) for key, val in tokenized_inputs.items()}

            # Forward pass
            outputs = self.model(**tokenized_inputs)
            embeddings = outputs.last_hidden_state
            if self.pooling == "mean":
                sequence_embeddings = self._masked_mean_pooling(embeddings, tokenized_inputs["attention_mask"])
            elif self.pooling == "cls":
                sequence_embeddings = embeddings[:, 0, :]
            else:
                raise ValueError(f"{self.pooling} is not a valid format for pooling selection. Please enter 'mean' for mean pooling, and 'cls' for cls token pooling")

            for k, v in zip(to_featurize, sequence_embeddings):
                self._store_in_cache(k, v.cpu())

    

if __name__ == "__main__":
    smiles = ["CCO", "C", "CH"]

    cbfeats = ChemBERTaFeaturizer(pooling="cls")
    feats = cbfeats.featurize(smiles)
    print(feats, len(feats))

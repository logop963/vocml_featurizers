import torch
import os
import numpy as np
import torch.nn as nn

from transformers import BertTokenizer, BertModel, AutoTokenizer

from vocml_featurizers.featurizer_manager import FeaturizerBaseClass
from vocml_featurizers.paths import OBP_CACHE


class ProtBERTFeaturizer(nn.Module, FeaturizerBaseClass):
    def __init__(self, pooling:str="mean", cache_file:str=os.path.join(OBP_CACHE, "ProtBERT-mapping.pkl")):
        super(ProtBERTFeaturizer, self).__init__()
        self.pooling = pooling
        # Initialize tokenizer & model and give cache dir
        cache_dir = os.path.join(OBP_CACHE, 'protBERTFeaturizer')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir=cache_dir)
        print("[ProtBERT] Loading model...")
        self.model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir=cache_dir)
        print("[ProtBERT] Model loaded successfully.")

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Identifiers
        self.data_column = 1
        self.name = 'ProtBERT'
        # Caching setup
        self.cache_file = os.path.join(OBP_CACHE, f"ProtBERT-mapping-{pooling}.pkl")
        self.cache = self._load_cache(self.cache_file)
        print(f"[ProtBERT] Cache file path: {cache_file}")
        print(f"[ProtBERT] Cache dir: {cache_dir}")
        print(f"[ProtBERT] Keys in loaded cache: {len(self.cache)}")


    def format_protein(self, seq: str) -> str:
        # Uppercase, strip whitespace, insert spaces between residues
        return " ".join(list(seq))

    def _tokenize(self, sequences):
        max_sequence_length = max([len(sequence) for sequence in sequences])
        formatted_sequences = []
        for seq in sequences:
            formatted_sequences.append(self.format_protein(seq))

        return self.tokenizer(formatted_sequences,
                              padding=True,
                              truncation=True,
                              max_length=370,
                              return_tensors="pt")
    
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
            elif self.pooling == None:
                sequence_embeddings = embeddings
            else:
                raise ValueError(f"{self.pooling} is not a valid format for pooling selection. Please enter 'mean' for mean pooling, and 'cls' for cls token pooling")

            for k, v in zip(to_featurize, sequence_embeddings):
                self._store_in_cache(k, v.cpu())


if __name__ == "__main__":
    from vbdata import m2or_data

    smiles = list(m2or_data()["mutated_sequence"].iloc[:5])
    cbfeats = ProtBERTFeaturizer(pooling="mean")
    toks = cbfeats._tokenize(smiles)
    
    print(cbfeats.cache)
    print(cbfeats.featurize(smiles))
    for tok in toks["input_ids"]:
        print(cbfeats.tokenizer.convert_ids_to_tokens(tok))
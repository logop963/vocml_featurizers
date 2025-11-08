'''
Toolset for managing featurization of semantic definitions of OBPs and VOCs
Goal is to define a base class that manages caching of features and carrying out of featurization logic
Subclasses should implement the featurization logic and the featurization manager should apply it
'''
import pickle

import numpy as np
import pickle
import os
import warnings
import torch
import tempfile
import shutil

class FeaturizerBaseClass():
    '''
    Tools for using featurizers compiled into an inherited class
    '''

    def _load_cache(self, cache_file: str):
        """Load cache from disk if available, else return empty dict."""
        if os.path.exists(cache_file):
            if os.path.getsize(cache_file) == 0:
                warnings.warn(f"Cache file {cache_file} is empty. Using an empty cache.", UserWarning)
                return {}

            print(f'Found cache for {self.name}!')
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load cache due to: {e}. Using an empty cache.", UserWarning)
                return {}

        warnings.warn(f"Cache file {cache_file} not found. Using an empty cache and computing features.", UserWarning)
        return {}

    def _save_cache(self, cache: dict, cache_file: str):
        """Save cache to disk safely using atomic file write."""
        tmp_dir = os.path.dirname(cache_file)
        with tempfile.NamedTemporaryFile("wb", dir=tmp_dir, delete=False) as tmp_file:
            pickle.dump(cache, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = tmp_file.name
        shutil.move(tmp_path, cache_file)

    def check_batch_for_cached(self, batch: np.ndarray, cache: dict[str, tuple]):
        """
        Returns list of samples not found in cache.
        """
        missing_samples = [s for s in batch if s not in cache and not None]
        return missing_samples

    def _store_in_cache(self, token: str, embedding: np.ndarray) -> None:
        self.cache[token] = embedding

    def featurize(self, input_data: np.ndarray) -> np.ndarray:
        """
        Featurizes peptide sequences on a batch level using caching.
        Args:
            input_data (torch.utils.data.DataLoader): Dataset to be featurized.
        Returns:
            np.ndarray: Featurized batch embeddings.
        """

        missing_samples = self.check_batch_for_cached(input_data, cache=self.cache)
        if missing_samples:
            self._featurization_logic(missing_samples)

        full_batch_embeddings = torch.stack([self.cache[key] for key in input_data])
        self._save_cache(self.cache, self.cache_file)

        return full_batch_embeddings

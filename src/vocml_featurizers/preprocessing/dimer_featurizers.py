from vocml_featurizers.preprocessing.base_wrapper import BaseVOCORFeaturizer
import torch

class VOCORConcatFeaturizer(BaseVOCORFeaturizer):
    def __init__(self, voc_encoder, or_encoder, voc_col, or_col, target_col, normalize, name):
        super().__init__(voc_encoder, or_encoder, voc_col, or_col, normalize)
        self.target_col = target_col
        self.name = name

    def __call__(self, batch):
        voc, or_ = super().__call__(batch)

        out = {
            "inputs": torch.cat([voc, or_], dim=1)
        }

        # handle single or multiple target columns
        if isinstance(self.target_col, (list, tuple)):
            for key in self.target_col:
                out[key] = batch[key]
        else:
            out[self.target_col] = batch[self.target_col]

        return out

class VOCORSeparateFeaturizer(BaseVOCORFeaturizer):
    def __init__(self, voc_encoder, or_encoder, voc_col, or_col, target_col, name):
        super().__init__(voc_encoder, or_encoder, voc_col, or_col)
        self.target_col = target_col
        self.name = name

    def __call__(self, batch):
        voc, or_ = super().__call__(batch)

        out = {
            "voc": voc.float(),
            "or": or_.float(),
        }

        # handle single or multiple target columns
        if isinstance(self.target_col, (list, tuple)):
            for key in self.target_col:
                out[key] = batch[key]
        else:
            out[self.target_col] = batch[self.target_col]

        return out

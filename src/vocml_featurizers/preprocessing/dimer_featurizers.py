from vocml_featurizers.preprocessing.base_wrapper import BaseVOCORFeaturizer
import torch

class VOCORConcatFeaturizer(BaseVOCORFeaturizer):
    def __init__(self, voc_encoder, or_encoder, voc_col, or_col, target_col, name):
        super().__init__(voc_encoder, or_encoder, voc_col, or_col)
        self.target_col = target_col
        self.name = name

    def __call__(self, batch):
        voc, or_ = super().__call__(batch)
        return {
            "inputs": torch.cat([voc, or_], dim=1),
            "targets": batch[self.target_col]
        }


class VOCORSeparateFeaturizer(BaseVOCORFeaturizer):
    def __init__(self, voc_encoder, or_encoder, voc_col, or_col, target_col, name):
        super().__init__(voc_encoder, or_encoder, voc_col, or_col)
        self.target_col = target_col
        self.name = name

    def __call__(self, batch):
        voc, or_ = super().__call__(batch)
        return {
            "voc": voc.float(),
            "or": or_.float(),
            "targets": batch[self.target_col]
        }

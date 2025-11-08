import torch.nn.functional as F
class BaseVOCORFeaturizer:
    """
    Base featurizer that applies featurizer operations to a dimer pair

    Params:
        - voc_encoder: VOC encoder object
        - or_encoder: OR encoder object
        - voc_id: column or key identifier for the VOC data
        - or_id: column or key identifier for the OR data
        - normalize: boolean to indicate doing normalization before passing downstream
    
    """
    def __init__(self, voc_encoder:object, or_encoder:object, voc_id:str, or_col:str, normalize:bool=None):
        self.voc_encoder = voc_encoder
        self.or_encoder = or_encoder
        self.voc_id = voc_id
        self.or_col = or_col
        self.normalize = normalize

    def featurize_voc(self, voc_input):
        if self.normalize:
            return F.normalize(self.voc_encoder.featurize(voc_input), p=2, dim=1)
        else:
            return(self.voc_encoder.featurize(voc_input))

    def featurize_or(self, or_input):
        if self.normalize:
            return F.normalize(self.or_encoder.featurize(or_input), p=2, dim=1)
        else:
            return(self.or_encoder.featurize(or_input))

    def __call__(self, batch):
        voc = self.featurize_voc(batch[self.voc_id])
        or_ = self.featurize_or(batch[self.or_col])
        return voc, or_
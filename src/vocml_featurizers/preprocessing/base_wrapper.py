import torch.nn.functional as F
class BaseVOCORFeaturizer:
    """
    Base featurizer that applies featurizer operations to a dimer pair

    Params:
        - voc_encoder: VOC encoder object
        - or_encoder: OR encoder object
        - voc_id: column or key identifier for the VOC data
        - or_id: column or key identifier for the OR data
        - normalize: boolean to indicate doing normalization before passing downstream, can pass tuple of bools for [VOC, OR] normalization respectively
    
    """
    def __init__(self, voc_encoder:object, or_encoder:object, voc_id:str, or_col:str, normalize:bool|tuple[bool]=None):
        self.voc_encoder = voc_encoder
        self.or_encoder = or_encoder
        self.voc_id = voc_id
        self.or_col = or_col
        self.normalize = normalize

    def featurize_voc(self, voc_input):
        voc_feats = self.voc_encoder.featurize(voc_input)
        if isinstance(self.normalize, bool):
            return F.normalize(voc_feats, p=2, dim=1) if self.normalize else voc_feats
        elif isinstance(self.normalize, tuple) and self.normalize[0]:
            return F.normalize(voc_feats, p=2, dim=1)
        return voc_feats

    def featurize_or(self, or_input):
        or_feats = self.or_encoder.featurize(or_input)
        if isinstance(self.normalize, bool):
            return F.normalize(or_feats, p=2, dim=1) if self.normalize else or_feats
        elif isinstance(self.normalize, tuple) and self.normalize[1]:
            return F.normalize(or_feats, p=2, dim=1)
        return or_feats

    def __call__(self, batch):
        voc = self.featurize_voc(batch[self.voc_id])
        or_ = self.featurize_or(batch[self.or_col])
        return voc, or_
"""Existing structural objectives backed by the hierarchical HIR-CodeBERT encoder."""

from lib_models.hircodebert import HIRCodeEncoder
from lib_models.hirgraphbert_pretrain import HIRGraphPretrainer


class HIRCodeBERTPretrainer(HIRGraphPretrainer):
    def __init__(self, config, weights=None):
        super().__init__(config, weights, encoder_cls=HIRCodeEncoder)

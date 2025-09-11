from __future__ import annotations
from dataclasses import dataclass
from ..ops import xp
@dataclass
class PCWeights:
    w_top: float=0.35; w_lat: float=0.15; w_bg: float=0.35; w_err: float=0.15
class PredictiveCodingMixer:
    def __init__(self, w: PCWeights=None): self.w=w or PCWeights()
    def mix(self, prior, role_emit, bigram_row, err):
        eps=1e-9
        prior=prior/(prior.sum()+eps); role_emit=role_emit/(role_emit.sum()+eps)
        bigram_row=bigram_row/(bigram_row.sum()+eps); err=err/(err.sum()+eps)
        logp = xp.log(prior+eps)*self.w.w_top + xp.log(role_emit+eps)*self.w.w_lat + xp.log(bigram_row+eps)*self.w.w_bg + xp.log(err+eps)*self.w.w_err
        return logp

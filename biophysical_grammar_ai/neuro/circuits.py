from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .neurons import PyramidalMC, PV_Basket, PV_Chandelier, SST_Martinotti, VIP
from .oscillations import CFC
@dataclass
class Layer:
    L2_3: List[PyramidalMC]; L5_6: List[PyramidalMC]; PV: List[PV_Basket]; CHAN: List[PV_Chandelier]; SST: List[SST_Martinotti]; VIP: List[VIP]
class LanguageNetwork:
    def __init__(self, n_pyr=12, n_inh=4):
        self.cfc = CFC()
        self.V1 = Layer([PyramidalMC() for _ in range(n_pyr)],[PyramidalMC() for _ in range(n_pyr)],[PV_Basket() for _ in range(n_inh)],[PV_Chandelier() for _ in range(1)], [SST_Martinotti() for _ in range(n_inh)], [VIP() for _ in range(n_inh)])
        self.STG= Layer([PyramidalMC() for _ in range(n_pyr)],[PyramidalMC() for _ in range(n_pyr)],[PV_Basket() for _ in range(n_inh)],[PV_Chandelier() for _ in range(1)], [SST_Martinotti() for _ in range(n_inh)], [VIP() for _ in range(n_inh)])
        self.MTG= Layer([PyramidalMC() for _ in range(n_pyr)],[PyramidalMC() for _ in range(n_pyr)],[PV_Basket() for _ in range(n_inh)],[PV_Chandelier() for _ in range(1)], [SST_Martinotti() for _ in range(n_inh)], [VIP() for _ in range(n_inh)])
        self.ATL= Layer([PyramidalMC() for _ in range(n_pyr)],[PyramidalMC() for _ in range(n_pyr)],[PV_Basket() for _ in range(n_inh)],[PV_Chandelier() for _ in range(1)], [SST_Martinotti() for _ in range(n_inh)], [VIP() for _ in range(n_inh)])
        self.IFG= Layer([PyramidalMC() for _ in range(n_pyr)],[PyramidalMC() for _ in range(n_pyr)],[PV_Basket() for _ in range(n_inh)],[PV_Chandelier() for _ in range(1)], [SST_Martinotti() for _ in range(n_inh)], [VIP() for _ in range(n_inh)])
    def rhythmic_gains(self,t=0.5): return self.cfc.gains(t)

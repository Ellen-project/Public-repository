from __future__ import annotations
from dataclasses import dataclass
from ..ops import xp, clip
@dataclass
class Channel:
    gmax: float; Erev: float; m: float=0.0; h: float=1.0; tau_m: float=1.0; tau_h: float=1.0; p_m: int=1; p_h: int=1
    def current(self,V): return self.gmax*(self.m**self.p_m)*(self.h**self.p_h)*(V-self.Erev)
    def step(self,V,dt): pass
class NaV(Channel):
    def __init__(self,gmax=120,Erev=50): super().__init__(gmax,Erev,0.0,1.0,0.2,1.0,3,1)
    def step(self,V,dt): m_inf=1/(1+xp.exp(-(V+35)/7)); h_inf=1/(1+xp.exp((V+58)/7)); self.m+=dt*(m_inf-self.m)/0.2; self.h+=dt*(h_inf-self.h)/1.0
class KV(Channel):
    def __init__(self,gmax=36,Erev=-77): super().__init__(gmax,Erev,0.0,1.0,3.0,1.0,4,0)
    def step(self,V,dt): m_inf=1/(1+xp.exp(-(V+28)/15)); self.m+=dt*(m_inf-self.m)/3.0
class KCa(Channel):
    def __init__(self,gmax=6.0,Erev=-80): super().__init__(gmax,Erev,0.0,1.0,5.0,1.0,1,0)
    def step(self,V,dt,Ca=0.1): m_inf=Ca/(Ca+0.3); self.m+=dt*(m_inf-self.m)/5.0
class Leak(Channel):
    def __init__(self,gmax=0.3,Erev=-65): super().__init__(gmax,Erev,1.0,1.0,1.0,1.0,1,0)
class CaV(Channel):
    def __init__(self,gmax=1.8,Erev=120): super().__init__(gmax,Erev,0.0,1.0,2.5,40.0,2,1)
    def step(self,V,dt): m_inf=1/(1+xp.exp(-(V+10)/6)); h_inf=1/(1+xp.exp((V+35)/6)); self.m+=dt*(m_inf-self.m)/2.5; self.h+=dt*(h_inf-self.h)/40.0
class TTypeCa(Channel):
    def __init__(self,gmax=2.0,Erev=120): super().__init__(gmax,Erev,0.0,1.0,7.0,60.0,2,1)
    def step(self,V,dt): m_inf=1/(1+xp.exp(-(V+60)/6)); h_inf=1/(1+xp.exp((V+80)/6)); self.m+=dt*(m_inf-self.m)/7.0; self.h+=dt*(h_inf-self.h)/60.0
class HCN(Channel):
    def __init__(self,gmax=1.0,Erev=-32): super().__init__(gmax,Erev,0.0,1.0,50.0,1.0,1,0)
    def step(self,V,dt): m_inf=1/(1+xp.exp((V+80)/8)); self.m+=dt*(m_inf-self.m)/50.0
class GIRK(Channel):
    def __init__(self,gmax=2.0,Erev=-95): super().__init__(gmax,Erev,0.0,1.0,40.0,1.0,1,0)
    def activate(self,level): self.m = clip(self.m + level, 0.0, 1.0)
    def step(self,V,dt): self.m += dt*(-self.m)/80.0

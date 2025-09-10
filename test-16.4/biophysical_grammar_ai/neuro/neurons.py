from __future__ import annotations
from dataclasses import dataclass
from ..ops import xp, clip
from .ion_channels import NaV, KV, KCa, Leak, CaV, HCN, GIRK
@dataclass
class SpikeRec: times: list
class PyramidalMC:
    def __init__(self):
        self.Cm=1.6
        self.V_soma=-70.0; self.V_basal=-72.0; self.V_api_p=-71.0; self.V_api_d=-74.0; self.V_AIS=-70.0
        self.Na_s=NaV(120); self.K_s=KV(36); self.Leak_s=Leak(0.3,-65); self.HCN_s=HCN(0.8); self.GIRK_s=GIRK(1.6)
        self.Na_b=NaV(40); self.K_b=KV(25); self.Leak_b=Leak(0.2,-70); self.Ca_b=CaV(1.2); self.KCa_b=KCa(4.0)
        self.Na_ap=NaV(30); self.K_ap=KV(20); self.Leak_ap=Leak(0.2,-68); self.Ca_ap=CaV(1.5); self.KCa_ap=KCa(5.0)
        self.Na_ad=NaV(20); self.K_ad=KV(18); self.Leak_ad=Leak(0.15,-68); self.Ca_ad=CaV(1.8); self.KCa_ad=KCa(6.0)
        self.g_sb=0.6; self.g_sp=0.5; self.g_pd=0.4; self.AIS_th=-50.5
        self.spikes = SpikeRec(times=[])
        self.Ca_soma=0.05; self.Ca_basal=0.05; self.Ca_api_p=0.05; self.Ca_api_d=0.05
        self.inputs_soma=[]; self.inputs_basal=[]; self.inputs_api_p=[]; self.inputs_api_d=[]
    def add_current(self, target, fn): getattr(self, f"inputs_{target}").append(fn)
    def _sumI(self, arr, t): return sum(float(fn(t)) for fn in arr) if arr else 0.0
    def _chan_step(self, V, dt, chans, Ca=None):
        I=0.0; I_Ca=0.0
        for ch in chans:
            if isinstance(ch, KCa): ch.step(V, dt, Ca if Ca is not None else 0.05)
            else: ch.step(V, dt)
            cur=ch.current(V); I+=cur
            if isinstance(ch, CaV): I_Ca += cur
        return I, I_Ca
    def _update_Ca(self, name, I_Ca, dt):
        Ca=getattr(self,name); k=1/150.0; alpha=0.003; Ca += dt*(-k*Ca + alpha*abs(I_Ca)); setattr(self,name,float(clip(Ca,0.0,10.0)))
    def step(self,t,dt):
        I_b,ICa_b=self._chan_step(self.V_basal,dt,[self.Na_b,self.K_b,self.Leak_b,self.Ca_b,self.KCa_b],self.Ca_basal)
        I_p,ICa_p=self._chan_step(self.V_api_p,dt,[self.Na_ap,self.K_ap,self.Leak_ap,self.Ca_ap,self.KCa_ap],self.Ca_api_p)
        I_d,ICa_d=self._chan_step(self.V_api_d,dt,[self.Na_ad,self.K_ad,self.Leak_ad,self.Ca_ad,self.KCa_ad],self.Ca_api_d)
        Iext_b=self._sumI(self.inputs_basal,t); Iext_p=self._sumI(self.inputs_api_p,t); Iext_d=self._sumI(self.inputs_api_d,t)
        I_sb=self.g_sb*(self.V_soma-self.V_basal); I_sp=self.g_sp*(self.V_soma-self.V_api_p); I_pd=self.g_pd*(self.V_api_p-self.V_api_d)
        self.V_basal += dt*((-I_b + I_sb + Iext_b)/self.Cm)
        self.V_api_p += dt*((-I_p + I_sp + Iext_p - I_pd)/self.Cm)
        self.V_api_d += dt*((-I_d + I_pd + Iext_d)/self.Cm)
        self._update_Ca("Ca_basal",ICa_b,dt); self._update_Ca("Ca_api_p",ICa_p,dt); self._update_Ca("Ca_api_d",ICa_d,dt)
        I_s,ICa_s=self._chan_step(self.V_soma,dt,[self.Na_s,self.K_s,self.Leak_s,self.HCN_s,self.GIRK_s],self.Ca_soma)
        I_csb=self.g_sb*(self.V_basal-self.V_soma); I_csp=self.g_sp*(self.V_api_p-self.V_soma); Iext_s=self._sumI(self.inputs_soma,t)
        self.V_soma += dt*((-I_s + I_csb + I_csp + Iext_s)/self.Cm); self.V_AIS = 0.85*self.V_soma + 0.10*self.V_api_p + 0.05*self.V_basal
        if self.V_AIS> -50.5: self.spikes.times.append(t); self.V_soma-=15.0; self.V_basal-=2.0; self.V_api_p-=2.0
        self._update_Ca("Ca_soma",ICa_s,dt)
class PV_Basket:
    def __init__(self): self.V_soma=-68.0; self.Na=NaV(90); self.K=KV(55); self.Leak=Leak(0.25,-65); self.Cm=0.9; self.spikes=[]
    def add_current(self, fn): self.ext=fn
    def step(self,t,dt):
        I=self.ext(t) if hasattr(self,"ext") else 0.0
        for ch in (self.Na,self.K,self.Leak): ch.step(self.V_soma,dt)
        self.V_soma += dt*((-self.Na.current(self.V_soma)-self.K.current(self.V_soma)-self.Leak.current(self.V_soma)+I)/self.Cm)
        if self.V_soma>-50.0: self.spikes.append(t); self.V_soma=-65.0
class PV_Chandelier(PV_Basket): pass
class SST_Martinotti:
    def __init__(self): self.V_soma=-67.0; self.Na=NaV(60); self.K=KV(35); self.Leak=Leak(0.25,-65); self.Cm=1.0; self.spikes=[]
    def add_current(self, fn): self.ext=fn
    def step(self,t,dt):
        I=self.ext(t) if hasattr(self,"ext") else 0.0
        for ch in (self.Na,self.K,self.Leak): ch.step(self.V_soma,dt)
        self.V_soma += dt*((-self.Na.current(self.V_soma)-self.K.current(self.V_soma)-self.Leak.current(self.V_soma)+I)/self.Cm)
        if self.V_soma>-50.0: self.spikes.append(t); self.V_soma=-65.0
class VIP:
    def __init__(self): self.V_soma=-66.0; self.Na=NaV(50); self.K=KV(30); self.Leak=Leak(0.20,-65); self.Cm=0.9; self.spikes=[]
    def add_current(self, fn): self.ext=fn
    def step(self,t,dt):
        I=self.ext(t) if hasattr(self,"ext") else 0.0
        for ch in (self.Na,self.K,self.Leak): ch.step(self.V_soma,dt)
        self.V_soma += dt*((-self.Na.current(self.V_soma)-self.K.current(self.V_soma)-self.Leak.current(self.V_soma)+I)/self.Cm)
        if self.V_soma>-50.0: self.spikes.append(t); self.V_soma=-65.0

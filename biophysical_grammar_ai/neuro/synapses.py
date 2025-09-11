from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..ops import xp, clip
@dataclass
class Receptor:
    gmax: float; Erev: float; s: float=0.0; tau_decay: float=5.0; alpha: float=0.5
    def open(self, weight=1.0): self.s = clip(self.s + self.alpha*weight, 0.0, 1.0)
    def step(self,V,dt): self.s += dt*(-self.s/max(self.tau_decay,1e-5))
    def current(self,V): return self.gmax*self.s*(V-self.Erev)
class AMPA(Receptor):
    def __init__(self,gmax=0.35,Erev=0.0): super().__init__(gmax,Erev,0.0,5.0,0.5)
class NMDA(Receptor):
    def __init__(self,gmax=0.25,Erev=0.0,Mg=1.0): super().__init__(gmax,Erev,0.0,100.0,0.15); self.Mg=Mg
    def mg_block(self,V): 
        try: import numpy as _np; return 1/(1 + self.Mg*_np.exp(-0.062*float(V))/3.57)
        except Exception: return 1/(1 + self.Mg*2.71828**(-0.062*float(V))/3.57)
    def current(self,V): return self.gmax*self.s*self.mg_block(V)*(V-self.Erev)
class GABA_A(Receptor):
    def __init__(self,gmax=0.55,Erev=-70.0): super().__init__(gmax,Erev,0.0,12.0,0.5)
class GABA_B(Receptor):
    def __init__(self,gmax=0.22,Erev=-95.0): super().__init__(gmax,Erev,0.0,150.0,0.1)
@dataclass
class STP:
    u: float=0.2; R: float=1.0; tau_rec: float=800.0; tau_facil: float=0.0
    def on_spike(self,dt):
        if self.tau_facil>0: self.u += (1-self.u)*(1-xp.exp(-dt/self.tau_facil))
        eff = self.u*self.R; self.R -= eff; self.R = max(0.0,float(self.R)); return eff
    def step(self,dt): self.R += (1-self.R)*(1-xp.exp(-dt/self.tau_rec)); self.R = clip(self.R,0.0,1.0)
class Synapse:
    def __init__(self, pre, post, kind="AMPA", weight=0.05, stp:Optional[STP]=None, target="basal"):
        self.pre=pre; self.post=post; self.kind=kind; self.weight=weight; self.target=target
        self.stp = stp or STP()
        if kind=="AMPA": self.r=AMPA()
        elif kind=="NMDA": self.r=NMDA()
        elif kind=="GABA_A": self.r=GABA_A()
        elif kind=="GABA_B": self.r=GABA_B()
        else: raise ValueError("unknown synapse")
        # plasticity params
        self.pre_trace=0.0; self.post_f=0.0; self.post_s=0.0
        self.tau_pre=20.0; self.tau_post_f=20.0; self.tau_post_s=120.0
        self.A_plus=0.010; self.A_minus=0.012; self.A_slow=0.006
        self.theta=0.4; self.tau_theta=1000.0; self.eta=1e-3
        self.w_min=0.0; self.w_max=0.3
    def _post_V(self):
        if hasattr(self.post, f"V_{self.target}"): return getattr(self.post, f"V_{self.target}")
        return getattr(self.post, "V_soma", -65.0)
    def on_pre_spike(self,dt): eff=self.stp.on_spike(dt); self.r.open(self.weight*eff); self.pre_trace += 1.0
    def on_post_spike(self,dt): self.post_f += 1.0; self.post_s += 1.0
    def step(self,dt):
        V = self._post_V(); self.stp.step(dt); self.r.step(V,dt)
        self.pre_trace *= 2.71828**(-dt/self.tau_pre)
        self.post_f    *= 2.71828**(-dt/self.tau_post_f)
        self.post_s    *= 2.71828**(-dt/self.tau_post_s)
        Ca = getattr(self.post, f"Ca_{self.target}", getattr(self.post,"Ca_soma",0.1))
        self.theta += dt*((Ca - self.theta)/self.tau_theta)
        ca_gate = (Ca - self.theta)
        dw = self.A_plus*self.post_f - self.A_minus*self.pre_trace + self.A_slow*self.post_s*self.pre_trace
        dw *= (1 + self.eta * ca_gate)
        self.weight = float(clip(self.weight + dw, self.w_min, self.w_max))
    def current(self):
        V=self._post_V(); return self.r.current(V)
class GapJunction:
    def __init__(self,a,b,g=0.2): self.a=a; self.b=b; self.g=g
    def current_ab(self): return self.g*(self.a.V_soma - self.b.V_soma)

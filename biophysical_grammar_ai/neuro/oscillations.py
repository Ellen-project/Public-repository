from __future__ import annotations
import math
class Osc:
    def __init__(self,f,amp=1.0,ph=0.0): self.f=f; self.amp=amp; self.ph=ph
    def value(self,t): return self.amp*math.sin(2*math.pi*self.f*t + self.ph)
class CFC:
    def __init__(self, theta=6.0,gamma=40.0,alpha=10.0,beta=20.0):
        self.theta=Osc(theta,1.0); self.gamma=Osc(gamma,1.0); self.alpha=Osc(alpha,1.0); self.beta=Osc(beta,1.0)
    def gains(self,t):
        th=self.theta.value(t); ga=self.gamma.value(t); al=self.alpha.value(t); be=self.beta.value(t)
        gamma_gain = 1.0 + max(0.0, th)*0.2 + max(0.0, al)*0.1
        beta_gain  = 1.0 + max(0.0, al)*0.2 + max(0.0, th)*0.05
        return dict(gamma=gamma_gain, beta=beta_gain, theta=th, alpha=al)

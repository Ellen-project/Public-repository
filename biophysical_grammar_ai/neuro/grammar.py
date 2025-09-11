from __future__ import annotations
import json, os
from typing import List, Dict
from ..ops import xp, clip, to_cpu
from .pcoding import PredictiveCodingMixer
from .synapses import Synapse
from .neurons import PyramidalMC
class RoleHMM:
    def __init__(self, V: int, roles: List[str]):
        self.roles=roles; self.R=len(roles); self.V=V
        self.T=xp.ones((self.R,self.R),dtype=xp.float32); self.E=xp.ones((self.R,V),dtype=xp.float32); self.pi=xp.ones((self.R,),dtype=xp.float32)
        self.dir_T=xp.ones_like(self.T)*0.5; self.dir_E=xp.ones_like(self.E)*0.5; self.dir_pi=xp.ones_like(self.pi)*0.5
    def _norm_rows(self,M): return M/(M.sum(axis=1, keepdims=True)+1e-8)
    def em_train(self, corpus: List[List[int]], iters=2):
        for _ in range(iters):
            A=xp.zeros_like(self.T); B=xp.zeros_like(self.E); P=xp.zeros_like(self.pi)
            for seq in corpus:
                N=len(seq); a=xp.zeros((N,self.R),dtype=xp.float32); a[0]=self.pi*self.E[:,seq[0]]; a[0]/=(a[0].sum()+1e-8)
                for t in range(1,N): a[t]=(a[t-1]@self.T)*self.E[:,seq[t]]; a[t]/=(a[t].sum()+1e-8)
                b=xp.zeros((N,self.R),dtype=xp.float32); b[-1]=1.0
                for t in range(N-2,-1,-1): b[t]=(self.T@(self.E[:,seq[t+1]]*b[t+1])); b[t]/=(b[t].sum()+1e-8)
                g=a*b; g/= (g.sum(axis=1, keepdims=True)+1e-8)
                xi=xp.zeros_like(self.T)
                for t in range(N-1):
                    x=(a[t][:,None]*self.T)*(self.E[:,seq[t+1]]*b[t+1])[None,:]
                    xi += x/(x.sum()+1e-8)
                P += g[0]; A += xi; 
                for t,w in enumerate(seq): B[:,w]+=g[t]
            self.T=self._norm_rows(A+self.dir_T); self.E=self._norm_rows(B+self.dir_E); self.pi=(P+self.dir_pi); self.pi/= (self.pi.sum()+1e-8)
    def role_mask(self, prev_role=None): return self.pi if prev_role is None else self.T[prev_role]
class GrammarNetwork:
    def __init__(self, vocab: List[str], save_path: str):
        self.vocab=vocab; self.V=len(vocab); self.idx={w:i for w,i in zip(vocab, range(len(vocab)))}
        self.save_path=save_path
        self.neurons = [PyramidalMC() for _ in range(self.V)]
        self.synapses = {} # (pre_idx, post_idx) -> Synapse object
        self.roles=["START","DET_DEF","DET_INDEF","DET_DEMON","ADJ_QUAL","ADJ_COMP","ADJ_SUPER","SUBJ","VERB","OBJ","MOD","PRT","AUX_ZEIT","AUX_MODAL","AUX_PERF","AUX_PROG","CONJ_COORD","CONJ_SUBORD","CONJ_CORREL","PREP_PLACE","PREP_TIME","PREP_MEANS","NUM","PUNCT","REL_PRON_R","REL_PRON_NR","REL_COMP_R","REL_COMP_NR","REL_ADV_R","REL_ADV_NR","END"]
        self.hmm=RoleHMM(self.V,self.roles)
        self.allowed_mask=xp.ones((self.V,),dtype=xp.float32)*0.0
        stop=set(["the","a","an","and","or","but","if","because","while","since","after","before","in","on","at","by","from","to","with","of",".",",","?","!",";","-","—","(",")"])
        self.content_mask=xp.ones((self.V,),dtype=xp.float32)
        for w,i in self.idx.items():
            if w in stop: self.content_mask[i]=0.20
        self.role_lex={"DET_DEF":["the"],"DET_INDEF":["a","an"],"DET_DEMON":["this","that","these","those"],
            "ADJ_QUAL":["robust","nuanced","concise","comprehensive","elegant","pragmatic","clear","precise"],
            "AUX_ZEIT":["am","is","are","was","were","be","being","been"],
            "AUX_MODAL":["will","would","should","can","could","may","might","must","shall"],
            "AUX_PERF":["have","has","had"],"AUX_PROG":["be","being"],
            "CONJ_COORD":["and","or","but","nor","yet","so"],"CONJ_SUBORD":["although","because","while","since","if","unless","after","before"],
            "PREP_PLACE":["in","on","at","over","under","between","among","within","through"],
            "PREP_TIME":["during","before","after","since","until","by","from","to"],"PREP_MEANS":["with","by","via","using","per"],
            "NUM":["one","two","three","ten","hundred","thousand"],"PUNCT":[".",",","?","!",";"],
            "PRT":["to","not"],"MOD":["however","therefore","furthermore","moreover","meanwhile","importantly","consequently"],
            "REL_PRON_R":["who","whom","whose","which"],"REL_COMP_R":["that"],"REL_ADV_R":["where","when","why"]}
        for r,toks in self.role_lex.items():
            if r in self.roles:
                rid=self.roles.index(r)
                for tok in toks:
                    if tok in self.idx: self.hmm.dir_E[rid, self.idx[tok]] += 40.0; self.allowed_mask[self.idx[tok]]=1.0
        for tok in ["we","you","i","it","this","that","data","model","result","analysis",".",",","?","!","patient","market","policy","risk","method"]:
            if tok in self.idx: self.allowed_mask[self.idx[tok]]=1.0
        if self.allowed_mask.sum()<10: self.allowed_mask[:]=1.0
        # Function words set for repetition control
        self._func_words = set(["the","a","an","and","or","but","if","because","while","since","after","before","in","on","at","by","from","to","with","of","we","you","i","it","this","that","these","those","be","is","are","am","was","were","do","does","did","have","has","had"])
        # Start-role prior: favor SUBJ -> AUX/VERB -> OBJ
        try:
            rid_subj = self.roles.index("SUBJ")
            self.hmm.pi *= 0.1
            self.hmm.pi[rid_subj] += 3.0
            self.hmm.pi = self.hmm.pi / (self.hmm.pi.sum()+1e-8)
        except Exception:
            pass

        self.pcm=PredictiveCodingMixer()
        self.E_emb=None; self.ctx_vec=None; self.lex_bias_ids=[]; self.lambda_copy=0.7
        self._role_bias_vec=None; self._role_lr_vec=None
        self.pre_trace=xp.zeros((self.V,),dtype=xp.float32); self.post_trace=xp.zeros((self.V,),dtype=xp.float32)
        self._det_roles=[self.roles.index(r) for r in self.roles if r.startswith("DET_")]
    def set_embeddings(self,E): self.E_emb=E
    def set_semantic_drive(self,vec,lex_ids): self.ctx_vec=vec; self.lex_bias_ids=[int(i) for i in lex_ids if 0<=int(i)<self.V]
    def allow_tokens(self,tokens):
        for t in tokens:
            if t in self.idx: self.allowed_mask[self.idx[t]]=1.0
    def set_context(self,diff,unc,domain_hint="default"): self.diff=diff; self.unc=unc; self.domain=domain_hint
    def set_domain_role_bias(self,role_bias:Dict,role_lr:Dict):
        self._role_bias_vec=xp.ones((len(self.roles),),dtype=xp.float32); self._role_lr_vec=xp.ones((len(self.roles),),dtype=xp.float32)
        for k,v in role_bias.items():
            if k in self.roles: self._role_bias_vec[self.roles.index(k)]=float(v)
        for k,v in role_lr.items():
            if k in self.roles: self._role_lr_vec[self.roles.index(k)]=float(v)
    def load(self):
        if os.path.exists(self.save_path):
            try:
                blob = json.load(open(self.save_path, "r", encoding="utf-8"))
            except Exception:
                return False
            if "W_sparse" in blob:
                self.synapses.clear()
                for i, pairs in enumerate(blob["W_sparse"]):
                    if i >= self.V: continue
                    for j, val in pairs:
                        if j < self.V:
                            pre_neuron = self.neurons[i]
                            post_neuron = self.neurons[j]
                            syn = Synapse(pre=pre_neuron, post=post_neuron, weight=float(val))
                            self.synapses[(i, j)] = syn
                            post_neuron.add_current(syn.target, syn.current)
                            pre_neuron.efferent_synapses.append(syn)
                            pre_neuron.efferent_synapses.append(syn)
            return True
        return False

    def save(self):
        W_sparse = [[] for _ in range(self.V)]
        for (pre, post), syn in self.synapses.items():
            if pre < self.V and post < self.V:
                W_sparse[pre].append([post, syn.weight])

        K = 12
        for i in range(len(W_sparse)):
            row = W_sparse[i]
            if not row: continue
            row.sort(key=lambda x: x[1], reverse=True)
            W_sparse[i] = row[:K]

        tmp = self.save_path + ".tmp"
        json.dump({"W_sparse": W_sparse}, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, self.save_path)

    def bootstrap_from_corpus(self, corpus: List[List[str]]):
        ids_corpus = []
        k = 8
        for sent in corpus:
            ids = [self.idx[w] for w in sent if w in self.idx]
            if not ids:
                continue
            ids_corpus.append(ids)
            for i, cur_id in enumerate(ids):
                for j in range(max(0, i - k), i):
                    pre_id = ids[j]
                    dist = i - j
                    if (pre_id, cur_id) not in self.synapses:
                        pre_neuron = self.neurons[pre_id]
                        post_neuron = self.neurons[cur_id]
                        syn = Synapse(pre=pre_neuron, post=post_neuron, weight=0.0)
                        self.synapses[(pre_id, cur_id)] = syn
                        post_neuron.add_current(syn.target, syn.current)
                        pre_neuron.efferent_synapses.append(syn)
                        pre_neuron.efferent_synapses.append(syn)
                    self.synapses[(pre_id, cur_id)].weight += 1.0 / dist

        weights_sum = {}
        for (pre, _), syn in self.synapses.items():
            weights_sum[pre] = weights_sum.get(pre, 0) + syn.weight
        for (pre, _), syn in self.synapses.items():
            if weights_sum.get(pre, 0) > 1e-6:
                syn.weight /= weights_sum[pre]

        if ids_corpus:
            self.hmm.em_train(ids_corpus, iters=50)

    def _length_hint(self):
        import numpy as np
        mu=18 + 8*getattr(self,'diff',0.2); sigma=5 + 4*getattr(self,'unc',0.2)
        return int(np.clip(np.random.normal(mu, sigma), 8, 72))

    def online_stdp(self, pre_id, post_id, role_id=None):
        self.pre_trace *= 0.95
        self.post_trace *= 0.95
        self.pre_trace[pre_id] += 1.0
        self.post_trace[post_id] += 1.0
        lr = 1.0
        if role_id is not None and self._role_lr_vec is not None:
            try:
                lr = float(self._role_lr_vec[role_id])
            except Exception:
                import numpy as _np
                lr = float(_np.asarray(to_cpu(self._role_lr_vec))[role_id])
        
        dw = lr * (0.02 * self.pre_trace[pre_id] * self.post_trace[post_id] - 0.01 * self.pre_trace.mean())

        if (pre_id, post_id) not in self.synapses:
            self.synapses[(pre_id, post_id)] = Synapse(pre=None, post=None, weight=0.0)
        
        syn = self.synapses[(pre_id, post_id)]
        syn.weight = clip(syn.weight + dw, 0.0, 1.0)

    def _prior_from_semantic(self):
        if self.E_emb is None or self.ctx_vec is None: return xp.ones((self.V,),dtype=xp.float32)
        p=self.E_emb @ self.ctx_vec; m=abs(p).max()+1e-9; p=p/m; p=p - p.min() + 1e-6
        if self.lex_bias_ids:
            for lid in self.lex_bias_ids:
                if 0<=lid< self.V: p[lid]+=0.4
        return p

    def next_token(self, cur_id, prev_role, recent_ids, freq):
        row = xp.zeros((self.V,), dtype=xp.float32)
        for (pre, post), syn in self.synapses.items():
            if pre == cur_id:
                row[post] = syn.weight

        if recent_ids and len(recent_ids) >= 2:
            prev1 = recent_ids[-2]
            row_prev1 = xp.zeros((self.V,), dtype=xp.float32)
            for (pre, post), syn in self.synapses.items():
                if pre == prev1:
                    row_prev1[post] = syn.weight
            row = 0.85 * row + 0.15 * row_prev1
        
        role_mask=self.hmm.role_mask(prev_role if prev_role is not None else 0)
        emit=self.hmm.E; role_probs=role_mask/(role_mask.sum()+1e-8)
        if getattr(self, "_role_bias_vec", None) is not None:
            rb=self._role_bias_vec; role_probs=role_probs*rb; role_probs/= (role_probs.sum()+1e-8)
        
        emit_mix=role_probs @ emit
        prior=self._prior_from_semantic(); err=self.content_mask.copy()
        logp=self.pcm.mix(prior, emit_mix, row, err)
        logp = logp + xp.log(self.allowed_mask + 1e-9)
        for wid,c in freq.items():
            if c>0:
                try:
                    tok=self.vocab[wid]
                    if tok in self._func_words:
                        logp[wid] -= 0.3*float(c)
                except Exception:
                    pass
            if 0<=wid<logp.size and c>0: logp[wid] -= 0.4*float(c)
        if recent_ids:
            for rid in recent_ids[-4:]:
                try:
                    tok=self.vocab[rid]
                    if tok in self._func_words:
                        logp[rid] -= 0.3
                except Exception:
                    pass
                if 0<=rid<logp.size: logp[rid]-=0.6
        x=logp/0.9; x=x - x.max(); p=xp.exp(x); p[cur_id]=0.0; p=p/(p.sum()+1e-9)
        order=xp.argsort(-p); csum=xp.cumsum(p[order]); cutoff=int((csum<=0.9).sum()); cutoff = cutoff if cutoff>=8 else min(16, p.size)
        mask=xp.zeros_like(p); mask[order[:cutoff]]=1.0; p=p*mask; p=p/(p.sum()+1e-9)
        resample_once = True
        if hasattr(p,"get"):
            import numpy as _np; probs = to_cpu(p)
            j=int(_np.random.choice(len(probs), p=probs))
            if resample_once and (j==cur_id or (recent_ids and j==recent_ids[-1])):
                probs[j] = 0.0; s = probs.sum();
                if s>1e-9: probs = probs/s; j=int(_np.random.choice(len(probs), p=probs))
        else:
            import numpy as _np; probs = to_cpu(p)
            j=int(_np.random.choice(len(probs), p=probs))
            if resample_once and (j==cur_id or (recent_ids and j==recent_ids[-1])):
                probs[j] = 0.0; s = probs.sum();
                if s>1e-9: probs = probs/s; j=int(_np.random.choice(len(probs), p=probs))
        r_next=int(xp.argmax(role_probs)); return j, r_next

    def generate(self,prompt_tokens: List[str], max_len=None):
        ids=[self.idx.get(tok,None) for tok in prompt_tokens if tok in self.idx]
        if not ids:
            for w in ["in","the","we","this"]:
                if w in self.idx: ids=[self.idx[w]]; break
            if not ids: ids=[int(self.V*0.05)]
        cur=ids[-1]; out=list(prompt_tokens); role=0; L=max_len or self._length_hint()
        punct={self.idx.get(tok) for tok in [".",",","?","!",";"] if tok in self.idx}
        freq={}; recent=list(ids[-4:])

        # Simulation parameters
        dt = 0.1  # ms
        sim_duration_ms = 5  # ms per token - keep it short as it's just a placeholder
        n_steps = int(sim_duration_ms / dt)

        while len(out) < L:
            # --- 1. Reset neuron states ---
            for neuron in self.neurons:
                neuron.spikes.times.clear()
                neuron.V_soma = -70.0; neuron.V_basal = -72.0; neuron.V_api_p = -71.0; neuron.V_api_d = -74.0

            # --- 2. Inject spike and run simulation ---
            current_neuron = self.neurons[cur]
            current_neuron.V_soma = 0  # Force spike

            first_spike_time = float('inf')
            winner_id = -1

            for i in range(n_steps):
                t = i * dt
                for idx, neuron in enumerate(self.neurons):
                    neuron.step(t, dt)
                    if winner_id == -1 and neuron.spikes.times and neuron.spikes.times[-1] <= t:
                        # Check if this is the first spike in the simulation
                        if t < first_spike_time:
                            first_spike_time = t
                            winner_id = idx
            
            # --- 3. Determine the winner ---
            if winner_id != -1:
                nxt = winner_id
            else:
                # Fallback: find neuron with the highest membrane potential
                max_v = -float('inf')
                nxt = -1
                for idx, neuron in enumerate(self.neurons):
                    if neuron.V_soma > max_v:
                        max_v = neuron.V_soma
                        nxt = idx
                if nxt == -1: nxt = (cur + 1) % self.V # Ultimate fallback

            # --- (Old logic replaced) ---
            # nxt,role=self.next_token(cur, role, recent, freq)
            role = 0 # Placeholder for role, as HMM is not driving decisions anymore

            # --- 4. Update state ---
            if cur in punct and nxt in punct: continue
            word=self.vocab[nxt]
            if len(out)>8 and word in (out[-1], out[-2] if len(out)>1 else ""): break
            out.append(word); freq[nxt]=freq.get(nxt,0)+1
            #self.online_stdp(cur, nxt, role_id=role)
            cur=nxt; recent.append(nxt); recent=recent[-4:]
            if word in (".","!","?") and len(out)>8: break
        
        text=" ".join(out); text=text.replace(" ,",",").replace(" .",".").replace(" !","!").replace(" ?","?").replace(" ;",";")
        if text and text[0].isalpha(): text=text[0].upper()+text[1:]
        if text[-1] not in ".!?": text+="."
        return text
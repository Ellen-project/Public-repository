from __future__ import annotations
import json, os
from typing import List, Dict
from ..ops import xp, clip, to_cpu
from .pcoding import PredictiveCodingMixer
from .snn_biophys import NeuronPopulation, SynapseGroup, SNN
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

        # SNN Backend Setup
        self.dt = 1e-4  # 0.1 ms timestep
        self.neurons = NeuronPopulation(N=self.V, dt=self.dt)
        self.synapses = [] # This will hold SynapseGroup objects
        self.net = None # This will be the SNN object
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

    def train_on_text(self, text: str):
        import cupy as cp
        import time
        import re

        # 1. Create a random, dense network to learn on
        print("[Training] Creating a random synaptic field...")
        num_synapses = int(self.V * self.V * 0.001)
        pre_ids = cp.random.randint(0, self.V, num_synapses, dtype=cp.int32)
        post_ids = cp.random.randint(0, self.V, num_synapses, dtype=cp.int32)
        initial_weights = cp.random.uniform(1e-10, 1e-9, num_synapses).astype(cp.float32)
        
        connections = []
        # This conversion to list of tuples is slow, but required by from_connections
        # A better from_connections would accept arrays directly.
        pre_ids_cpu = cp.asnumpy(pre_ids)
        post_ids_cpu = cp.asnumpy(post_ids)
        weights_cpu = cp.asnumpy(initial_weights)
        for i in range(num_synapses):
            connections.append((pre_ids_cpu[i], post_ids_cpu[i], weights_cpu[i]))

        syn_group = SynapseGroup.from_connections(
            pre=self.neurons,
            post=self.neurons,
            connections=connections,
            receptor_types='AMPA+NMDA', # NMDA is crucial for plasticity
            name="learned_synapses"
        )
        self.synapses.append(syn_group)
        self.net = SNN(neurons=[self.neurons], synapses=self.synapses)
        print(f"[Training] Network created with {syn_group.E} synapses.")

        # 2. Prepare text for training
        # Simple text cleaning
        text = re.sub(r'\s+', ' ', text) # Collapse whitespace
        text = re.sub(r'\[[0-9]+\]', '', text) # Remove reference numbers like [1]
        tokens = [self.idx[tok] for tok in text.lower().split() if tok in self.idx]
        if not tokens:
            print("[Training] No valid tokens found in text to learn from.")
            return

        # 3. Run the learning simulation
        print(f"[Training] Starting learning loop on {len(tokens)} tokens...")
        I_inject = 4e-9 # 4 nA
        sim_steps_per_token = 100 # ms to simulate for each word
        
        start_time = time.time()
        for i, token_id in enumerate(tokens):
            Iext = {self.neurons: cp.zeros(self.V, dtype=cp.float32)}
            Iext[self.neurons][token_id] = I_inject

            for _ in range(sim_steps_per_token):
                self.net.step(Iext)
            
            # Report progress
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[Training] ... processed {i+1}/{len(tokens)} tokens in {elapsed:.2f}s.")
        
        total_time = time.time() - start_time
        print(f"[Training] Finished training on {len(tokens)} tokens in {total_time:.2f}s.")
    def load(self):
        if not os.path.exists(self.save_path):
            return False
        try:
            blob = json.load(open(self.save_path, "r", encoding="utf-8"))
        except Exception:
            return False

        if "W_sparse" in blob:
            connections = []
            for i, pairs in enumerate(blob["W_sparse"]):
                if i >= self.V: continue
                for j, val in pairs:
                    if j < self.V:
                        # Scale weight to be a conductance value (e.g., in nano-Siemens)
                        connections.append((i, j, float(val) * 5e-9))

            if not connections:
                return True # File exists but no synapses

            syn_group = SynapseGroup.from_connections(
                pre=self.neurons,
                post=self.neurons,
                connections=connections,
                receptor_types='AMPA',
                name="loaded_synapses"
            )
            self.synapses.append(syn_group)
            self.net = SNN(neurons=[self.neurons], synapses=self.synapses)

        return True

    def save(self):
        if not self.synapses:
            return

        # Assume we are saving the first synapse group
        syn_group = self.synapses[0]

        # --- Bridge Logic: Convert SNN_BIPHYS format back to sparse list ---
        W_sparse = [[] for _ in range(self.V)]
        
        # Move data from GPU to CPU for processing
        import cupy as cp
        pre_ids = cp.asnumpy(syn_group.pre_idx)
        post_ids = cp.asnumpy(syn_group.post_idx)
        weights = cp.asnumpy(syn_group.w)

        for i in range(syn_group.E):
            pre = int(pre_ids[i])
            post = int(post_ids[i])
            weight = weights[i] / 1e-8 # Rescale back
            if pre < self.V and post < self.V:
                W_sparse[pre].append([post, weight])

        K = 12 # Top-K connections to save
        for i in range(len(W_sparse)):
            row = W_sparse[i]
            if not row: continue
            row.sort(key=lambda x: x[1], reverse=True)
            W_sparse[i] = row[:K]

        tmp = self.save_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"W_sparse": W_sparse}, f)
            os.replace(tmp, self.save_path)
        except Exception as e:
            print(f"Error saving grammar file: {e}")

    def bootstrap_from_corpus(self, corpus: List[List[str]]):
        ids_corpus = []
        k = 8

        temp_synapses = {}
        for sent in corpus:
            ids = [self.idx[w] for w in sent if w in self.idx]
            if not ids:
                continue
            ids_corpus.append(ids)
            for i, cur_id in enumerate(ids):
                for j in range(max(0, i - k), i):
                    pre_id = ids[j]
                    dist = i - j
                    if (pre_id, cur_id) not in temp_synapses:
                        temp_synapses[(pre_id, cur_id)] = 0.0
                    temp_synapses[(pre_id, cur_id)] += 1.0 / dist

        weights_sum = {}
        for (pre, _), w in temp_synapses.items():
            weights_sum[pre] = weights_sum.get(pre, 0) + w
        for (pre, cur), w in temp_synapses.items():
            if weights_sum.get(pre, 0) > 1e-6:
                temp_synapses[(pre, cur)] = w / weights_sum[pre]

        # --- Use the new factory method ---
        connections = []
        for (pre, post), w in temp_synapses.items():
            # Scale weight to be a conductance value (e.g., in nano-Siemens)
            connections.append((pre, post, w * 5e-9))

        if not connections:
            print("Warning: No synapses created from corpus.")
            return

        syn_group = SynapseGroup.from_connections(
            pre=self.neurons,
            post=self.neurons,
            connections=connections,
            receptor_types='AMPA',
            name="corpus_synapses"
        )

        self.synapses.append(syn_group)
        self.net = SNN(neurons=[self.neurons], synapses=self.synapses)

        if ids_corpus:
            self.hmm.em_train(ids_corpus, iters=50)

    def _length_hint(self):
        import numpy as np
        mu=18 + 8*getattr(self,'diff',0.2); sigma=5 + 4*getattr(self,'unc',0.2)
        return int(np.clip(np.random.normal(mu, sigma), 8, 72))

    def online_stdp(self, pre_id, post_id, role_id=None):
        # This is now handled by the SynapseGroup's internal plasticity mechanisms.
        pass

    def _prior_from_semantic(self):
        if self.E_emb is None or self.ctx_vec is None: return xp.ones((self.V,),dtype=xp.float32)
        p=self.E_emb @ self.ctx_vec; m=abs(p).max()+1e-9; p=p/m; p=p - p.min() + 1e-6
        if self.lex_bias_ids:
            for lid in self.lex_bias_ids:
                if 0<=lid< self.V: p[lid]+=0.4
        return p

    def next_token(self, cur_id, prev_role, recent_ids, freq):
        # This is now handled by the SNN simulation in the generate method.
        pass

    def generate(self,prompt_tokens: List[str], max_len=None):
        if not self.net:
            print("Error: SNN network not initialized.")
            return ""

        ids=[self.idx.get(tok,None) for tok in prompt_tokens if tok in self.idx]
        if not ids:
            for w in ["in","the","we","this"]:
                if w in self.idx: ids=[self.idx[w]]; break
            if not ids: ids=[int(self.V*0.05)]
        
        cur=ids[-1]
        out=list(prompt_tokens)
        L=max_len or self._length_hint()
        punct={self.idx.get(tok) for tok in [".",",","?","!",";"] if tok in self.idx}
        freq={}
        recent=list(ids[-4:])

        import cupy as cp
        import time
        sim_duration_ms = 50  # ms
        n_steps = int(sim_duration_ms / (self.dt * 1000))
        I_inject = 3e-9 # 3 nA

        while len(out) < L:
            # --- 1. Reset network state (voltages, spikes etc.) ---
            for group in self.net.neurons:
                group.V[:] = group.p.EL
                group.w[:] = 0
                group.VT[:] = group.p.VT0
                group.ref_count[:] = 0
                group.spike[:] = 0
            for group in self.net.synapses:
                group.s[:] = 0
                group.r[:] = 0

            # --- 2. Run simulation to find the next token ---
            winner_id = -1
            
            # Inject current into the 'cur' neuron for a short duration
            Iext = {self.neurons: cp.zeros(self.V, dtype=cp.float32)}
            Iext[self.neurons][cur] = I_inject

            # --- Time measurement ---
            is_first_token = len(out) == len(prompt_tokens)
            total_step_time = 0.0

            for i in range(n_steps):
                if is_first_token:
                    start_time = time.time()

                spikes_out = self.net.step(Iext)
                
                if is_first_token:
                    total_step_time += time.time() - start_time

                # Turn off injection after a few steps
                if i > 5:
                    Iext[self.neurons][cur] = 0.0

                # Check for a winner (first spike other than the input)
                spikes = cp.asnumpy(spikes_out[self.neurons])
                fired_indices = spikes.nonzero()[0]
                if len(fired_indices) > 0:
                    for fired_idx in fired_indices:
                        if fired_idx != cur:
                            winner_id = fired_idx
                            break
                if winner_id != -1:
                    break
            
            if is_first_token:
                avg_step_time_ms = (total_step_time / n_steps) * 1000
                print(f"[Performance] Avg. time per SNN step: {avg_step_time_ms:.4f} ms")

            # --- 3. Determine the winner ---
            if winner_id != -1:
                nxt = winner_id
            else:
                # Fallback: find neuron with highest membrane potential
                max_v = -float('inf')
                potentials = cp.asnumpy(self.neurons.V)
                nxt = int(potentials.argmax())
                if nxt == cur: # Avoid getting stuck
                    nxt = int(potentials.argsort()[-2]) # Take second highest

            # --- 4. Update state ---
            if cur in punct and nxt in punct: continue
            word=self.vocab[nxt]
            if len(out)>8 and word in (out[-1], out[-2] if len(out)>1 else ""): break
            out.append(word); freq[nxt]=freq.get(nxt,0)+1
            
            # Placeholder for new plasticity call
            # self.net.synapses[0].phase1_and_plasticity(self.neurons.spike)

            cur=nxt; recent.append(nxt); recent=recent[-4:]
            if word in (".","!","?") and len(out)>8: break
        
        text=" ".join(out); text=text.replace(" ,",",").replace(" .",".").replace(" !","!").replace(" ?","?").replace(" ;",";")
        if text and text[0].isalpha(): text=text[0].upper()+text[1:]
        if text[-1] not in ".!?": text+="."
        return text
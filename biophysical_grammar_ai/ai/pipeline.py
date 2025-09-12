from __future__ import annotations
from typing import List
from ..neuro.circuits import LanguageNetwork
from ..neuro.grammar import GrammarNetwork
from ..ops import xp
from .vocab import build_vocab, build_embeddings
DOMAINS = {"law":["court","legal","evidence","policy","regulation","due","contract","liability","precedent","jurisdiction"],
"medicine":["patient","clinical","diagnosis","symptom","therapy","treatment","dose","adverse","risk"],
"finance":["market","portfolio","investment","equity","return","volatility","hedge","liquidity","credit"],
"education":["student","teacher","curriculum","assessment","learning","pedagogy","school","classroom"],
"security":["attack","threat","encryption","vulnerability","protocol","access","defense","incident"],
"sports":["team","season","tournament","coach","score","defense","offense","match","league"],
"tech":["algorithm","dataset","benchmark","pipeline","token","embedding","latency","scalability"]}
DOMAIN_ROLE_BIAS={"default":{"ADJ_QUAL":1.05,"CONJ_COORD":1.05},"law":{"CONJ_COORD":1.30,"CONJ_SUBORD":1.25,"REL_PRON_R":1.25,"REL_COMP_R":1.20,"AUX_MODAL":1.10},"medicine":{"NUM":1.30,"ADJ_QUAL":1.15,"PREP_TIME":1.10},"finance":{"NUM":1.35,"PREP_MEANS":1.15,"CONJ_COORD":1.05},"education":{"ADJ_QUAL":1.20,"CONJ_SUBORD":1.15},"security":{"MOD":1.20,"PREP_MEANS":1.10},"sports":{"AUX_ZEIT":1.15,"CONJ_COORD":1.20},"tech":{"ADJ_QUAL":1.10,"NUM":1.10,"PREP_MEANS":1.10}}
DOMAIN_ROLE_LR={"default":{"ADJ_QUAL":1.05},"law":{"CONJ_COORD":1.25,"CONJ_SUBORD":1.20,"REL_PRON_R":1.15},"medicine":{"NUM":1.25,"ADJ_QUAL":1.10},"finance":{"NUM":1.30,"PREP_MEANS":1.10},"education":{"CONJ_SUBORD":1.15},"security":{"MOD":1.20},"sports":{"AUX_ZEIT":1.10},"tech":{"ADJ_QUAL":1.10,"NUM":1.10}}
DOMAIN_PARAMS={"default":{"length_mu":18.0,"length_sigma":6.0,"rhythm":{"theta":6.0,"gamma":40.0,"alpha":10.0,"beta":20.0}},"law":{"length_mu":28.0,"length_sigma":8.0,"rhythm":{"theta":5.0,"gamma":35.0,"alpha":9.5,"beta":18.0}},"medicine":{"length_mu":24.0,"length_sigma":7.0,"rhythm":{"theta":6.0,"gamma":38.0,"alpha":10.0,"beta":20.0}},"finance":{"length_mu":22.0,"length_sigma":6.0,"rhythm":{"theta":6.0,"gamma":42.0,"alpha":10.5,"beta":22.0}},"education":{"length_mu":20.0,"length_sigma":6.0,"rhythm":{"theta":6.5,"gamma":36.0,"alpha":9.0,"beta":18.0}},"security":{"length_mu":18.0,"length_sigma":5.0,"rhythm":{"theta":5.5,"gamma":45.0,"alpha":11.0,"beta":20.0}},"sports":{"length_mu":18.0,"length_sigma":5.0,"rhythm":{"theta":6.0,"gamma":40.0,"alpha":9.5,"beta":18.0}},"tech":{"length_mu":19.0,"length_sigma":5.0,"rhythm":{"theta":5.5,"gamma":44.0,"alpha":10.5,"beta":22.0}}}
def infer_domain(tokens: List[str]) -> str:
    s=set(tokens); best=("default",0)
    for dom,kws in DOMAINS.items():
        c=len([k for k in kws if k in s])
        if c>best[1]: best=(dom,c)
    return best[0]
class Pipeline:
    def __init__(self, data_dir: str):
        self.net=LanguageNetwork(n_pyr=12, n_inh=4)
        self.vocab=build_vocab(target_size=3000)
        self.emb=build_embeddings(self.vocab, dim=256)
        self.gn=GrammarNetwork(self.vocab, save_path=f"{data_dir}/grammar.json")
        self.gn.set_embeddings(self.emb)
        tpls=["Hello , and thanks for your message .","Here is a concise answer to your question .","In short , we can explain the steps clearly .","If you prefer , I can provide more detail in the next turn .","From a legal standpoint , the contract should specify scope and remedies .","Clinically , the patient improved after the initial therapy .","The portfolio balances equity exposure with hedges that limit volatility .","The curriculum integrates formative assessment with useful feedback .","A layered control strategy reduces the blast radius of an incident .","The team improved possession by adjusting tempo and spacing .","The pipeline validates data quality before model training ."]
        if not self.gn.load():
            print("[Training] No pre-trained grammar found. Starting self-learning...")
            learning_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
            
            try:
                import requests
                from bs4 import BeautifulSoup

                print(f"[Training] Fetching content from {learning_url}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                }
                response = requests.get(learning_url, headers=headers)
                response.raise_for_status() # Raise an exception for bad status codes
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text from the main content area of Wikipedia
                content_div = soup.find(id='mw-content-text')
                if content_div:
                    # Remove tables, nav boxes, etc.
                    for tag in content_div.find_all(['table', '.navbox']):
                        tag.decompose()
                    fetched_text = content_div.get_text(separator=' ', strip=True)
                else:
                    fetched_text = soup.get_text(separator=' ', strip=True)

                print("[Training] Content fetched successfully.")
                self.gn.train_on_text(fetched_text)

            except Exception as e:
                print(f"[Training] Failed to fetch or train: {e}")
                # Fallback to bootstrap if web fetch fails
                self.gn.bootstrap_from_corpus([["hello","world"]])

            print("[Training] Learning complete. Saving knowledge...")
            self.gn.save()
            print("[Training] Knowledge saved to grammar.json.")

        print(f"[Network Info] Neurons: {self.gn.neurons.N}, Synapses: {self.gn.synapses[0].E if self.gn.synapses else 'N/A'}")
    def sense_encode(self, text: str):
        difficulty=min(1.0, max(0.0, len(text)/140.0)); uncertainty=0.6 if "?" in text else 0.25; return difficulty, uncertainty
    def _content_tokens(self, toks):
        SW=set(["the","a","an","and","or","but","if","because","while","since","after","before","in","on","at","by","from","to","with","of","we","you","they","i","it","this","that","these","those"])
        return [t for t in toks if len(t)>2 and t not in SW and t.isalpha()]
    def _semantic_context(self, tokens):
        idx=[self.gn.idx[t] for t in tokens if t in self.gn.idx]
        if not idx: return None, []
        import numpy as np
        vec=self.emb[idx].mean(axis=0); vec = vec / (np.linalg.norm(vec) + 1e-9); return vec, idx[:12]
    def respond(self, text: str) -> str:
        tokens=text.strip().lower().split()
        difficulty, uncertainty=self.sense_encode(text)
        domain=infer_domain(tokens)
        params=DOMAIN_PARAMS.get(domain, DOMAIN_PARAMS['default'])
        try:
            self.net.cfc.theta.f=params["rhythm"]["theta"]; self.net.cfc.gamma.f=params["rhythm"]["gamma"]; self.net.cfc.alpha.f=params["rhythm"]["alpha"]; self.net.cfc.beta.f=params["rhythm"]["beta"]
        except Exception: pass
        self.gn.set_context(difficulty, uncertainty, domain_hint=domain)
        rb=DOMAIN_ROLE_BIAS.get(domain, DOMAIN_ROLE_BIAS['default']); rl=DOMAIN_ROLE_LR.get(domain, DOMAIN_ROLE_LR['default']); self.gn.set_domain_role_bias(rb, rl)
        content=self._content_tokens(tokens); vec, salient=self._semantic_context(content)
        if vec is not None: self.gn.set_semantic_drive(vec, salient)
        self.gn.allow_tokens([t for t in tokens if t in self.gn.idx])
        prompt=[t for t in tokens if t in self.gn.idx][-20:]
        if "?" in text: prompt=["here","is","a","concise","answer"] + prompt
        return self.gn.generate(prompt, max_len=None)
    

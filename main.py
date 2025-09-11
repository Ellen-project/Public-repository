import os
from biophysical_grammar_ai.ai.pipeline import Pipeline
from biophysical_grammar_ai import ops
def main():
    base=os.path.join(os.path.dirname(__file__), "biophysical_grammar_ai","data"); os.makedirs(base, exist_ok=True)
    pipe=Pipeline(data_dir=base)
    print("[Biophysical-Grammar AI 3f2eac61f9e1205ada39d1fccfc8c8938b771a55] Ready. Type your message (Ctrl+C to exit)."); print("(Backend:", ops.reason, ")")
    while True:
        try:
            text=input("\nYou: ").strip()
            if not text: continue
            reply=pipe.respond(text); print("AI :", reply)
        except (KeyboardInterrupt, EOFError):
            print("\nBye."); break
if __name__=="__main__": main()
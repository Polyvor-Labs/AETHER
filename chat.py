import torch
import pickle
import time
import os
import re
from config import Config
from model.architecture import Aether1Model

def generate():
    cfg = Config()
    if not os.path.exists(cfg.MODEL_PATH):
        print("❌ Model weights not found. Please run main.py first.")
        return

    # Load Vocab
    with open(cfg.VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    cfg.VOCAB_SIZE = len(vocab)
    stoi = { w:i for i,w in enumerate(vocab) }
    itos = { i:w for i,w in enumerate(vocab) }
    
    # Load Model
    model = Aether1Model(cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()
    
    print("🤖 Assistant is Online! (Type 'bye' to exit)")
    print("------------------------------------------------")
    
    while True:
        try:
            txt = input("\nYou: ")
            if txt.lower() in ['exit', 'quit', 'bye']: break
            
            parts = txt.split()
            temp = 0.6 
            try:
                if len(parts) > 1 and 0.1 <= float(parts[-1]) <= 2.0:
                    temp = float(parts[-1])
                    txt = " ".join(parts[:-1])
            except: pass

            # Format Prompt
            prompt = f"User: {txt}\nAssistant:"
            clean_prompt = re.sub(r'([.,!?;:()"])', r' \1 ', prompt)
            
            ids = [stoi.get(w, 0) for w in clean_prompt.split()]
            if not ids: ids = [0]
            ctx = torch.tensor([ids], dtype=torch.long, device=cfg.DEVICE)
            
            print(f"Assistant: ", end="", flush=True)
            
            response_words = []
            
            with torch.no_grad():
                for _ in range(100): 
                    cond = ctx[:, -cfg.SEQ_LEN:]
                    out, _, _ = model(cond)
                    
                    logits = out[:, -1, :] / temp
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    
                    word = itos[next_id]
                    
                    # --- STOP TOKEN LOGIC ---
                    stop_words = ["User:", "Assistant:", "<|end|>", "User", "Assistant"]
                    if word in stop_words: 
                        break
                    
                    # Prevent Repetition
                    if len(response_words) > 2:
                        if word == response_words[-1] == response_words[-2]: break

                    ctx = torch.cat([ctx, torch.tensor([[next_id]], device=cfg.DEVICE)], dim=1)
                    response_words.append(word)
                    
                    # Clean Output Formatting
                    if word in [".", ",", "!", "?", ":"]:
                        print(word, end="", flush=True)
                    else:
                        print(" " + word, end="", flush=True)
                    time.sleep(0.03)
            print()
        except KeyboardInterrupt: break

if __name__ == "__main__":
    generate()
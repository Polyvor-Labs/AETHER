import torch
import time
import os
import pickle
import json
import re
from collections import Counter
from config import Config
from model.architecture import Aether1Model

def load_chat_dataset(path, max_vocab=30000):
    if not os.path.exists(path):
        print(f"❌ Error: File '{path}' not found! Please upload dataset.json.")
        exit()

    print(f"📚 Loading Dataset: {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f: 
            chat_data = json.load(f)
    except Exception as e:
        print(f"❌ JSON Error: {e}")
        exit()

    full_text = ""
    for item in chat_data:
        q = item.get('question', '')
        a = item.get('answer', '')

        full_text += f"User: {q}\nAssistant: {a}\n<|end|>\n"

    print("🧹 Cleaning & Tokenizing...")

    text = re.sub(r'([.,!?;:()"])', r' \1 ', full_text)
    words = text.replace('\n', ' \n ').split()

    count = Counter(words)
    most_common = count.most_common(max_vocab)
    unique_words = [w for w, c in most_common]

    special_tokens = ["<UNK>", "User:", "Assistant:", "<|end|>"]
    for t in special_tokens:
        if t not in unique_words: unique_words.append(t)

    print(f"✅ Final Vocab Size: {len(unique_words)}")

    stoi = { w:i for i,w in enumerate(unique_words) }
    unk_id = stoi["<UNK>"]
    encoded = [stoi.get(w, unk_id) for w in words]

    return torch.tensor(encoded, dtype=torch.long), unique_words

def get_batch(data, config):
    ix = torch.randint(len(data) - config.SEQ_LEN, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i+config.SEQ_LEN] for i in ix])
    y = torch.stack([data[i+1:i+config.SEQ_LEN+1] for i in ix])
    return x.to(config.DEVICE), y.to(config.DEVICE)

def train():
    cfg = Config()
    data, unique_words = load_chat_dataset(cfg.DATASET_PATH, max_vocab=cfg.MAX_VOCAB)
    cfg.VOCAB_SIZE = len(unique_words)

    print(f"🌌 AETHER (Assistant Mode) Initialized on: {cfg.DEVICE}")
    model = Aether1Model(cfg).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() 

    print("🚀 Starting Training...")
    model.train()

    steps_per_epoch = 200 
    print(f"📅 Training Plan: {cfg.EPOCHS} Epochs x {steps_per_epoch} Steps")

    for epoch in range(cfg.EPOCHS):
        start = time.time()
        losses = 0

        for _ in range(steps_per_epoch):
            inputs, targets = get_batch(data, cfg)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                _, loss, _ = model(inputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses += loss.item()

        avg_loss = losses / steps_per_epoch
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")

    print("💾 Saving Model...")
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    with open(cfg.VOCAB_PATH, 'wb') as f: pickle.dump(unique_words, f)
    print("✅ Training Complete. Model saved.")

if __name__ == "__main__":
    train()
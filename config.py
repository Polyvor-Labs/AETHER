import torch

class Config:
    # --- Model Architecture ---
    VOCAB_SIZE = 0         # Set automatically during data loading
    D_MODEL = 512          # Dimension of the model (High capacity)
    NUM_LAYERS = 6         # Number of hybrid layers
    DROPOUT = 0.3          # High dropout to prevent memorization
    
    # --- Experimental Features ---
    ADAPTIVE_THRESHOLD = 0.5 
    PONDER_PENALTY = 0.01
    HEBBIAN_LR = 0.01      # Plasticity learning rate
    HEBBIAN_DECAY = 0.99   # Plasticity decay factor
    
    # --- Training Config ---
    BATCH_SIZE = 16        # Tuned for T4 GPU stability
    SEQ_LEN = 64           # Context window length
    LEARNING_RATE = 1e-4   # Slow and steady learning
    EPOCHS = 20            # Optimal point to avoid overfitting on small datasets
    
    MAX_VOCAB = 30000      # Vocabulary limit
    
    # Paths
    MODEL_PATH = "aether_weights.pth"
    VOCAB_PATH = "aether_vocab.pkl"
    DATASET_PATH = "dataset.json"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
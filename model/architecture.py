import torch
import torch.nn as nn
from .layers import AdaptiveLayer, HebbianPlasticityHead

class Aether1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.SEQ_LEN, config.D_MODEL) * 0.02)
        
        self.layers = nn.ModuleList([
            AdaptiveLayer(config.D_MODEL) for _ in range(config.NUM_LAYERS)
        ])
        
        self.final_norm = nn.LayerNorm(config.D_MODEL)
        self.head = HebbianPlasticityHead(
            config.D_MODEL, 
            config.VOCAB_SIZE, 
            eta=config.HEBBIAN_LR, 
            decay=config.HEBBIAN_DECAY
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embedding(idx)
        
        # Add positional encoding
        if T <= self.config.SEQ_LEN: 
            x = x + self.pos_encoding[:, :T, :]
        
        ponder_cost = 0
        for layer in self.layers:
            x, think = layer(x)
            ponder_cost += think
            
        x = self.final_norm(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Reshape for CrossEntropy: [Batch*Seq, Vocab]
            main_loss = loss_fn(logits.reshape(-1, self.config.VOCAB_SIZE), targets.reshape(-1))
            loss = main_loss + (self.config.PONDER_PENALTY * ponder_cost)
            
        return logits, loss, ponder_cost
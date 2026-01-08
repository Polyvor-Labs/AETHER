# 🌌 AETHER: Adaptive Hebbian SSM

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research_Prototype-blue?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**AETHER** (**A**daptive **E**ngine with **T**emporal **H**ebbian **E**volution & **R**ecurrence) is an experimental text generation model built **entirely from scratch** in PyTorch. 

It challenges the current dominance of Transformer architectures by exploring a **Hybrid State Space Model (SSM)** combined with biologically inspired **Hebbian Plasticity**. Unlike standard LLMs with static weights, AETHER features "living weights" that adapt dynamically during inference to capture short-term context.

---

## 🧠 Core Technical Innovations

AETHER diverges from the standard GPT (Decoder-only Transformer) architecture in three key areas:

### 1. Hybrid SSM Backbone (The Body)
Instead of $O(N^2)$ Self-Attention, AETHER utilizes a **Gated Convolutional Recurrent** mechanism.
* **Complexity:** $O(N)$ (Linear scaling with sequence length).
* **Mechanism:** It uses depth-wise 1D convolutions to capture local context and gated linear units (GLU) to modulate information flow, inspired by Mamba/S4 architectures.

### 2. Hebbian Plasticity Layer (The Memory)
The final output layer implements **Online Hebbian Learning** (*"Cells that fire together, wire together"*).
* **Concept:** The model maintains a `hebbian_trace` matrix that updates during the forward pass.
* **Function:** This acts as a Short-Term Working Memory, allowing the model to "remember" the immediate conversation context without modifying the frozen long-term weights.
* **Update Rule:** $\Delta W_{fast} = \eta \cdot (y_{out} \otimes x_{in})$

### 3. Adaptive Computation (The Logic)
Includes an experimental **Router** mechanism R(x) within each layer.
* **Function:** Determines the "thinking intensity" for each token. In future iterations, this will allow the model to dynamically exit early for simple tokens (saving compute) or ponder longer for complex queries.

---

## 📊 Research Findings & Observations

This project was developed through iterative experimentation. Below are the findings documented during the training process on a Tesla T4 GPU (Chatbot Dataset ~1.2k pairs):

| Phase | Observation | Analysis |
| :--- | :--- | :--- |
| **Phase 1: Overfitting** | Loss dropped to `<0.03` in 5 epochs. Model repeated user prompts verbatim. | The model capacity (`d_model=512`) was too large for the small dataset, leading to instant memorization. **Fix:** Increased Dropout to `0.3` and reduced Epochs. |
| **Phase 2: "Dreamy" State** | Model produced grammatically perfect sentences but semantically drifted (e.g., associating "Camera" with "Fertilizers"). | The SSM backbone successfully learned syntax, but the lack of a massive pre-training corpus caused semantic hallucination. |
| **Phase 3: Stable Assistant** | Coherent, persona-aligned responses. | Achieved by implementing **Stop-Token Logic** in the inference engine to prevent identity leakage (e.g., the model writing "User:" tags recursively). |

---

## 📂 Project Structure

```bash
AETHER/
├── model/
│   ├── __init__.py
│   ├── layers.py          # SSM Blocks, Hebbian Plasticity Head, Router
│   └── architecture.py    # Main Model Assembly
├── main.py                # Training Script (JSON Loader & Training Loop)
├── chat.py                # Inference Engine (Chat Interface)
├── config.py              # Hyperparameters (Tuned for T4 GPU)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the required packages.


```bash
git clone https://github.com/Polyvor-Labs/AETHER.git
cd AETHER
```
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Create a file named dataset.json in the root directory. The trainer expects a list of JSON objects:


```json
[
  {
    "question": "Hello!",
    "answer": "Hi! How can I help you today?"
  },
  {
    "question": "What is your name?",
    "answer": "I am the Assistant."
  }
]
```

### 3. Training
Run the training script. This will tokenize your data, build the vocabulary, and train the model.

```bash
python main.py
```

Artifacts aether_weights.pth and aether_vocab.pkl will be saved locally.

### 4. Inference (Chat)
Run the interactive chat script to test the model.

```bash
python chat.py
```

### ⚙️ Configuration (config.py)

You can tune the model performance by modifying config.py:

- D_MODEL: (Default: 512) Size of the vector representation. Higher = Smarter, but slower.

- NUM_LAYERS: (Default: 6) Depth of the network.

- EPOCHS: (Default: 20) The sweet spot found during research to prevent overfitting on small datasets.

- DROPOUT: (Default: 0.3) Regularization to force the model to learn patterns, not memorize text.

### 🔮 Future Roadmap

- [ ] Scale Up: Train on the TinyStories dataset (~1GB text) to test true generalization.

- [ ] Flash Attention: Integrate optimized kernels for faster training.

- [ ] RLHF: Implement Reinforcement Learning from Human Feedback to align responses.

## 📜 License
Distributed under the MIT License. See LICENSE for more information.

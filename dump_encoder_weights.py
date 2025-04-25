import torch
import numpy as np
from train import SAE  # or wherever your SAE class is defined

# Config — adjust if your dim or filename differs
INPUT_DIM = 384
HIDDEN_DIM = 500
model = SAE(INPUT_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("sae.pt"))
model.eval()

# Extract encoder weights
W = model.enc.weight.detach().cpu().numpy()  # shape (hidden_dim, input_dim)
b = model.enc.bias.detach().cpu().numpy()    # shape (hidden_dim,)

# Save as .npz
np.savez("encoder_weights.npz", W=W, b=b)
print("✅ Saved encoder_weights.npz")

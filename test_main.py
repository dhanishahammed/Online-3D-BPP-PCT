import torch
from attention_model import AttentionModel
from envs.packing_env import PackingEnv
from tools.utils import observation_decode_leaf_node

# ====== Parameters (adjust based on your setup) ======
embedding_dim = 128
hidden_dim = 128
n_encode_layers = 2
internal_node_holder = 5
internal_node_length = 16
leaf_node_holder = 10

# ====== Initialize model ======
model = AttentionModel(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    n_encode_layers=n_encode_layers,
    internal_node_holder=internal_node_holder,
    internal_node_length=internal_node_length,
    leaf_node_holder=leaf_node_holder
)

# Put model in eval mode
model.eval()

# ====== Initialize environment ======
# You might need to replace this with an actual environment setup
env = PackingEnv()

# Reset environment to get initial observation
obs = env.reset()
obs_tensor = torch.tensor(obs).unsqueeze(0).float()  # Add batch dim if needed

# ====== Forward pass ======
with torch.no_grad():
    action_log_prob, pointers, dist_entropy, hidden, dist = model(obs_tensor)

# ====== Print outputs ======
print("Action Log Prob:", action_log_prob)
print("Selected Action (Pointer):", pointers)
print("Entropy:", dist_entropy)
print("Hidden (context):", hidden.shape)

print("Input shape:", input.shape)
print("Embedding shape:", embeddings.shape)

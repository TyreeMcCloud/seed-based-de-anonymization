import os
import networkx as nx
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
from scipy.spatial import distance

data_folder = "data"

# Load Graphs
G1 = nx.read_edgelist(os.path.join(data_folder, "seed_G1.edgelist"), nodetype=int)
G2 = nx.read_edgelist(os.path.join(data_folder, "seed_G2.edgelist"), nodetype=int)

# Load Seed Mapping
seed_mapping = pd.read_csv(os.path.join(data_folder, "seed_mapping_test.txt"), sep=" ", header=None, names=["G1", "G2"])

# Step 1: Generate Embeddings using DeepWalk-style Random Walks
def generate_walks(graph, num_walks=5, walk_length=10):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if neighbors:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks

def build_vocab(walks):
    vocab = defaultdict(int)
    for walk in walks:
        for node in walk:
            vocab[node] += 1
    return vocab

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.output_layer = nn.Linear(embed_size, vocab_size)  # Ensure output is same as vocab size

    def forward(self, x):
        x = self.embeddings(x)
        return self.output_layer(x)  # Output layer ensures proper shape

def train_embeddings(graph, embed_size=128, epochs=10, lr=0.01, batch_size=128):
    walks = generate_walks(graph)
    vocab = build_vocab(walks)
    node_to_idx = {node: i for i, node in enumerate(vocab.keys())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    model = SkipGram(len(vocab), embed_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  

    print(f"Vocabulary size: {len(vocab)}")

    for epoch in range(epochs):
        total_loss = 0
        batch_center = []
        batch_target = []
        
        for walk in walks:
            for i, node in enumerate(walk):
                center_idx = node_to_idx[node]
                target_idx = random.choice(list(node_to_idx.values()))

                batch_center.append(center_idx)
                batch_target.append(target_idx)

                # Process batch
                if len(batch_center) >= batch_size:
                    center_tensor = torch.tensor(batch_center, dtype=torch.long)
                    target_tensor = torch.tensor(batch_target, dtype=torch.long)

                    optimizer.zero_grad()
                    logits = model(center_tensor)
                    loss = loss_fn(logits, target_tensor)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_center = []  # Clear batch
                    batch_target = []

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    embeddings = {idx_to_node[i]: model.embeddings.weight[i].detach().numpy() for i in range(len(vocab))}
    return embeddings

# Generate embeddings for both graphs
embeddings_G1 = train_embeddings(G1)
embeddings_G2 = train_embeddings(G2)

# Convert seed mapping into training data
X_train = np.array([embeddings_G1[g1] for g1 in seed_mapping["G1"]])
y_train = np.array([embeddings_G2[g2] for g2 in seed_mapping["G2"]])

# Step 2: Define a Neural Network for Learning the Mapping
class MappingNN(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=256):
        super(MappingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = MappingNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Training Loop
epochs = 2
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Step 4: Predict Matches for Unmapped Nodes
unmapped_G1 = set(G1.nodes()) - set(seed_mapping["G1"])
unmapped_G1_features = np.array([embeddings_G1[node] for node in unmapped_G1])

unmapped_G1_tensor = torch.tensor(unmapped_G1_features, dtype=torch.float32)
predicted_G2_embeddings = model(unmapped_G1_tensor).detach().numpy()

# Match predicted embeddings to closest G2 node
unmapped_G2 = set(G2.nodes()) - set(seed_mapping["G2"])
unmapped_G2_embeddings = np.array([embeddings_G2[node] for node in unmapped_G2])
unmapped_G2_list = list(unmapped_G2)

final_mapping = {**dict(zip(seed_mapping["G1"], seed_mapping["G2"]))}

for i, g1_node in enumerate(unmapped_G1):
    if not unmapped_G2_list:  # If unmapped_G2_list is empty, stop the loop
        print("Warning: No more unmapped G2 nodes available.")
        break

    # Compute distances
    distances = [distance.euclidean(predicted_G2_embeddings[i], emb) for emb in unmapped_G2_embeddings]
    
    # Get the best match index
    best_match_idx = np.argmin(distances)

    # Validate index before using it
    if 0 <= best_match_idx < len(unmapped_G2_list):
        best_match = unmapped_G2_list[best_match_idx]
        final_mapping[g1_node] = best_match
        unmapped_G2_list.remove(best_match)  # Remove matched node
    else:
        print(f"Skipping {g1_node}: Invalid best_match_idx {best_match_idx}")

print("Neural Network-based de-anonymization completed. Mapping process finished.")

# Save final_mapping to a file
output_file = "Solution2.txt"

with open(output_file, "w") as f:
    for g1, g2 in final_mapping.items():
        f.write(f"{g1} {g2}\n")

print(f"Mapping saved successfully to {output_file}")
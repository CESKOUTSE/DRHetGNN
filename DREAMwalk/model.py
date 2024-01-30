import torch
from torch_geometric.data import Data
import torch.nn as nn
import pickle

# Chargement du graphe
def load_graph(filename):
    
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    print(graph)
        
    return graph

# Couches MLP
def mlp(input_size, hidden_size):
    
    liner1 = nn.Linear(input_size, hidden_size)
    liner2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(x):
        x = liner1(x)
        x = nn.ReLU()(x) 
        x = liner2(x)
        return x

    return forward

# Noeuds drug
def drug_node(input_size, hidden_size):
    
    func = mlp(input_size, hidden_size)
    
    def forward(x):
        return func(x)

    return forward

# Noeuds disease  
def disease_node(input_size, hidden_size):
    
    func = mlp(input_size, hidden_size)
    
    def forward(x):  
        return func(x)
    
    return forward

# Forward pass du modèle
def model_forward(data, input_size, hidden_size):
    
    drug_features = data.x['drug']    
    disease_features = data.x['disease']
    
    drug_embeddings = drug_node(input_size, hidden_size)
    disease_embeddings = disease_node(input_size, hidden_size)
    
    concatenated = torch.cat([...])  
    
    outputs = nn.Linear(concatenated)
    
    return outputs
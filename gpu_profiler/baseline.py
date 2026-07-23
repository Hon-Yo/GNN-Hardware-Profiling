import torch
from torch.nn import Linear, Sequential, ReLU, Embedding
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, 
    GINConv, 
    GATConv, 
    PNAConv, 
    global_add_pool
)
from torch_geometric.utils import degree
from torch_geometric.transforms import AddRandomWalkPE

# Import your custom profiler
from profiler import GNNProfiler

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================

class GCNBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch_ptr=None, pe=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GINBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        mlp1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        mlp2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)

    def forward(self, x, edge_index, batch_ptr=None, pe=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GATBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, batch_ptr=None, pe=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class PNABaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, deg):
        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.conv1 = PNAConv(in_channels, hidden_channels, aggregators=aggregators, scalers=scalers, deg=deg)
        self.conv2 = PNAConv(hidden_channels, out_channels, aggregators=aggregators, scalers=scalers, deg=deg)

    def forward(self, x, edge_index, batch_ptr=None, pe=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GIN_VN_Baseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # The Virtual Node requires its own embedding (state)
        self.vn_emb = Embedding(1, hidden_channels)
        
        # NEW: Project initial node features to match hidden_channels
        self.node_proj = Linear(in_channels, hidden_channels)
        
        # Standard GIN MLPs (Updated mlp1 to expect hidden_channels)
        mlp1 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        mlp2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        
        # An MLP specifically to update the Virtual Node state between layers
        self.vn_mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))

    def forward(self, x, edge_index, batch_ptr, pe=None):
        # NEW: Map raw features (size 1) to hidden_channels (size 64)
        h = self.node_proj(x)
        
        # Initialize virtual node
        vn_state = self.vn_emb(torch.zeros(1, dtype=torch.long, device=x.device))
        
        # --- Layer 1 ---
        # Broadcast virtual node state to all real nodes
        h = h + vn_state[0] 
        # Standard Message Passing
        h = self.conv1(h, edge_index).relu() 
        # Update virtual node by pooling all real nodes into it
        vn_state = vn_state + global_add_pool(h, batch_ptr)
        vn_state = self.vn_mlp(vn_state)
        
        # --- Layer 2 ---
        h = h + vn_state[0]
        h = self.conv2(h, edge_index)
        
        return h

class DGNBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pe_dim=20):
        super().__init__()
        # DGN requires positional encodings concatenated to the node features
        # We increase the input dimension to account for the Random Walk PE
        self.linear_in = Linear(in_channels + pe_dim, hidden_channels)
        
        # We use a dense message passing layer to simulate the directional aggregation
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch_ptr=None, pe=None):
        # Concatenate features with Positional Encodings
        if pe is not None:
            x = torch.cat([x, pe], dim=-1)
        
        x = self.linear_in(x).relu()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ==========================================
# 2. DATASET LOADER UTILITY
# ==========================================

def get_dataloader(dataset_name="ZINC", batch_size=1):
    print(f"Loading {dataset_name} dataset with Positional Encodings...")
    transform = AddRandomWalkPE(walk_length=20, attr_name='pe')
    
    if dataset_name == "ZINC":
        # Changed pre_transform to transform
        dataset = ZINC(root="data/ZINC", subset=True, split="val", transform=transform) 
    elif dataset_name == "QM9":
        # Changed pre_transform to transform
        dataset = QM9(root="data/QM9", transform=transform)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader

def compute_pna_degrees(dataset):
    print("Computing degree histogram for PNA...")
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================

def main():
    dataset, loader = get_dataloader(dataset_name="ZINC", batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ZINC specific dimensions
    in_channels = dataset.num_features if dataset.num_features > 0 else 1
    
    # Pre-compute degrees for PNA
    deg = compute_pna_degrees(dataset)
    
    # Dictionary of all 6 models
    models = {
        "GCN": GCNBaseline(in_channels, 64, dataset.num_classes).to(device),
        "GIN": GINBaseline(in_channels, 64, dataset.num_classes).to(device),
        "GAT": GATBaseline(in_channels, 64, dataset.num_classes).to(device),
        "PNA": PNABaseline(in_channels, 64, dataset.num_classes, deg).to(device),
        "GIN-VN": GIN_VN_Baseline(in_channels, 64, dataset.num_classes).to(device),
        "DGN": DGNBaseline(in_channels, 64, dataset.num_classes, pe_dim=20).to(device)
    }

    for model_name, model in models.items():
        print(f"\nPreparing {model_name}...")
        model.eval()
        
        # 1. WARM-UP PHASE
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                x = batch.x.float() if batch.x.dtype != torch.float32 else batch.x
                # Notice we pass batch.batch and batch.pe explicitly now
                _ = model(x, batch.edge_index, batch_ptr=batch.batch, pe=getattr(batch, 'pe', None))
                if i >= 10: 
                    break
        
        # 2. INJECT PROFILER
        profiler = GNNProfiler(model, model_name)
        
        # 3. PROFILED INFERENCE LOOP
        print(f"Running profiled streaming inference for {model_name}...")
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i <= 10: continue 
                
                batch = batch.to(device)
                x = batch.x.float() if batch.x.dtype != torch.float32 else batch.x
                
                # Signal the profiler that a new graph is starting
                profiler.start_graph()
                
                # Forward pass
                out = model(x, batch.edge_index, batch_ptr=batch.batch, pe=getattr(batch, 'pe', None))
                
                # Signal the profiler to log the metrics for this specific graph
                profiler.end_graph(batch)
                
                if i >= 510: 
                    break
        
        # 4. PRINT RESULTS AND EXPORT
        profiler.print_results()

        profiler.export_to_csv("zinc_hardware_baseline.csv")

if __name__ == "__main__":
    main()
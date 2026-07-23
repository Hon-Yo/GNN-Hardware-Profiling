import time
import torch
import csv 
import os
from torch_geometric.nn import MessagePassing

class GNNProfiler:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        
        # Global cumulative stats
        self.stats = {"NT_time": 0.0, "MP_time": 0.0, "layer_calls": 0}
        
        # Per-graph tracking for streaming analysis
        self.graph_log = []
        
        # Internal state trackers for the current active graph
        self._current_graph_mp = 0.0
        self._current_graph_total = 0.0
        
        self._attach_hooks()

    def _attach_hooks(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                self._wrap_layer(module)

    def _wrap_layer(self, layer):
        original_forward = layer.forward
        original_propagate = layer.propagate

        def profiled_propagate(*args, **kwargs):
            t0 = time.perf_counter()
            result = original_propagate(*args, **kwargs)
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            layer._temp_mp_time += elapsed
            self._current_graph_mp += elapsed  # Track for the current graph
            return result

        def profiled_forward(*args, **kwargs):
            layer._temp_mp_time = 0.0 
            
            t0 = time.perf_counter()
            result = original_forward(*args, **kwargs)
            t1 = time.perf_counter()
            
            total_time = t1 - t0
            mp_time = layer._temp_mp_time
            nt_time = total_time - mp_time
            
            # Accumulate global statistics
            self.stats["MP_time"] += mp_time
            self.stats["NT_time"] += nt_time
            self.stats["layer_calls"] += 1
            
            # Accumulate total layer time for this specific graph
            self._current_graph_total += total_time
            
            return result

        layer.propagate = profiled_propagate
        layer.forward = profiled_forward

    def start_graph(self):
        """Call this right before a new graph enters the model."""
        self._current_graph_mp = 0.0
        self._current_graph_total = 0.0

    def end_graph(self, batch_data):
        """Call this right after the model finishes processing a graph."""
        total_latency = self._current_graph_total
        mp_latency = self._current_graph_mp
        nt_latency = total_latency - mp_latency
        
        # Prevent division by zero if a layer is instantaneous or empty
        bottleneck_ratio = (mp_latency / nt_latency) if nt_latency > 0 else 0.0
        
        # Extract complexity metrics from the PyG Batch object
        num_nodes = batch_data.num_nodes
        num_edges = batch_data.num_edges
        
        self.graph_log.append({
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "total_latency_ms": total_latency * 1000, # Convert to milliseconds
            "nt_latency_ms": nt_latency * 1000,
            "mp_latency_ms": mp_latency * 1000,
            "bottleneck_ratio_mp_to_nt": bottleneck_ratio
        })

    def print_results(self):
        nt = self.stats['NT_time']
        mp = self.stats['MP_time']
        total = nt + mp
        
        print(f"\n{'='*40}")
        print(f" PROFILING SUMMARY: {self.model_name}")
        print(f"{'='*40}")
        print(f"Graphs Streamed   : {len(self.graph_log)}")
        print(f"Total Layer Calls : {self.stats['layer_calls']}")
        print(f"Total Layer Time  : {total:.6f} sec")
        print("-" * 40)
        
        if total > 0:
            nt_pct = (nt / total) * 100
            mp_pct = (mp / total) * 100
            print(f"Node Transform (NT) : {nt:.6f} sec ({nt_pct:.1f}%)")
            print(f"Message Passing (MP): {mp:.6f} sec ({mp_pct:.1f}%)")
            
            # Print average complexity of streamed graphs
            avg_nodes = sum(g['num_nodes'] for g in self.graph_log) / len(self.graph_log)
            avg_edges = sum(g['num_edges'] for g in self.graph_log) / len(self.graph_log)
            print(f"Avg Graph Complexity: {avg_nodes:.1f} nodes, {avg_edges:.1f} edges")
        print(f"{'='*40}\n")

    def export_to_csv(self, filename="profiling_results.csv"):
        """Exports the streaming graph data to a CSV file. Appends if the file already exists."""
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row only if the file was just created
            if not file_exists:
                writer.writerow([
                    "Model", 
                    "Graph_ID", 
                    "Num_Nodes", 
                    "Num_Edges", 
                    "Total_Latency_ms", 
                    "NT_Latency_ms", 
                    "MP_Latency_ms", 
                    "Bottleneck_Ratio"
                ])
            
            # Loop through the logged graphs and write each as a row
            for i, graph in enumerate(self.graph_log):
                writer.writerow([
                    self.model_name,
                    i + 1,  # Graph_ID (1-indexed for readability)
                    graph['num_nodes'],
                    graph['num_edges'],
                    f"{graph['total_latency_ms']:.6f}",
                    f"{graph['nt_latency_ms']:.6f}",
                    f"{graph['mp_latency_ms']:.6f}",
                    f"{graph['bottleneck_ratio_mp_to_nt']:.6f}"
                ])
                
        print(f"  --> Successfully exported {len(self.graph_log)} rows to {filename}")
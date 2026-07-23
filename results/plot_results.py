import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plot(cpu_csv, gpu_csv):
    # 1. Load the telemetry data
    try:
        cpu_df = pd.read_csv(cpu_csv)
        gpu_df = pd.read_csv(gpu_csv)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Column names based on your telemetry output
    model_col = 'Model'
    nt_col = 'NT_Latency_ms'
    mp_col = 'MP_Latency_ms'

    # Group by 'Model' and calculate the mean
    cpu_grouped = cpu_df.groupby(model_col, sort=False)[[nt_col, mp_col]].mean().reset_index()
    gpu_grouped = gpu_df.groupby(model_col, sort=False)[[nt_col, mp_col]].mean().reset_index()

    # Define the exact desired order for the models
    desired_order = ['GIN', 'GIN-VN', 'GCN', 'GAT', 'PNA', 'DGN']

    # Apply the specific ordering using reindex
    cpu_grouped = cpu_grouped.set_index(model_col).reindex(desired_order).reset_index()
    gpu_grouped = gpu_grouped.set_index(model_col).reindex(desired_order).reset_index()

    models = cpu_grouped[model_col].tolist()
    
    # Extract the averaged data arrays
    cpu_nt = cpu_grouped[nt_col].values
    cpu_mp = cpu_grouped[mp_col].values
    gpu_nt = gpu_grouped[nt_col].values
    gpu_mp = gpu_grouped[mp_col].values

    # 2. Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Average GNN Streaming Latency: CPU vs. GPU', fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.6

    # 3. Plot CPU Data (Left)
    ax1.bar(x, cpu_nt, width, label='Node Transformation (NT)', color='#1f77b4')
    ax1.bar(x, cpu_mp, width, bottom=cpu_nt, label='Message Passing (MP)', color='#ff7f0e')
    ax1.set_title('CPU Execution', fontsize=14)
    ax1.set_ylabel('Average Latency (ms)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # ACADEMIC STANDARD LEGEND: 
    # Placed once on the left plot where the reader begins.
    ax1.legend(loc='upper left', fontsize=11)

    # 4. Plot GPU Data (Right)
    ax2.bar(x, gpu_nt, width, label='Node Transformation (NT)', color='#1f77b4')
    ax2.bar(x, gpu_mp, width, bottom=gpu_nt, label='Message Passing (MP)', color='#ff7f0e')
    ax2.set_title('GPU Execution', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('hardware_comparison_chart.png', dpi=300)
    print("Successfully generated and saved 'hardware_comparison_chart.png'")
    plt.show()

if __name__ == "__main__":
    generate_comparison_plot('FINAL_averaged_telemetry_cpu.csv', 'FINAL_averaged_telemetry_gpu.csv')
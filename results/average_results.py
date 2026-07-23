import pandas as pd
import glob

def main():
    print("Finding CSV files...")
    # This finds all files that match the pattern (e.g., zinc_hardware_baseline_1.csv)
    # Ensure your 5 files are in the same directory and have somewhat similar names!
    files = glob.glob("zinc_hardware_baseline*.csv") 
    
    if len(files) == 0:
        print("No CSV files found! Check your file names.")
        return
        
    print(f"Found {len(files)} files. Merging and calculating averages...")
    
    # Read all CSVs and concatenate them into one massive 15,000 row table
    df_list = [pd.read_csv(f) for f in files]
    df_all = pd.concat(df_list)

    # ---------------------------------------------------------
    # OUTPUT 1: The "Per-Graph" Average (3,000 rows)
    # This averages the 5 runs together for Graph #1, Graph #2, etc.
    # ---------------------------------------------------------
    df_graph_avg = df_all.groupby(["Model", "Graph_ID", "Num_Nodes", "Num_Edges"]).mean().reset_index()
    df_graph_avg.to_csv("FINAL_averaged_telemetry.csv", index=False)
    print(" -> Saved 'FINAL_averaged_telemetry.csv' (Use this for your scatter plots)")

    # ---------------------------------------------------------
    # OUTPUT 2: The "Paper Summary" Table (6 rows)
    # This takes the averages of everything to give you exactly 1 row per model
    # ---------------------------------------------------------
    df_model_avg = df_graph_avg.groupby(["Model"]).mean().reset_index()
    df_model_avg = df_model_avg.drop(columns=["Graph_ID"]) # Graph ID doesn't make sense for a global average
    df_model_avg.to_csv("FINAL_paper_summary.csv", index=False)
    print(" -> Saved 'FINAL_paper_summary.csv' (Use this for your final paper/presentation)")

if __name__ == "__main__":
    main()
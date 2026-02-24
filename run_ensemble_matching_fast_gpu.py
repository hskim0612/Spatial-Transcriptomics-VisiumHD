import os
import torch
import torch_directml
import scanpy as sc
import numpy as np
import pandas as pd
import gc
import time

# [ENGINEERING STANDARDS]
DEVICE = torch_directml.device()
SAMPLE_ID = "13604-RE-1"
ST_PATH = "E:/20251117_Visium/CellRanger/13604-RE-1/binned_outputs/square_008um/filtered_feature_bc_matrix.h5"
POS_PATH = "E:/20251117_Visium/CellRanger/13604-RE-1/binned_outputs/square_008um/spatial/tissue_positions.parquet"
REF_PATH = "Rebecca_WT_Full_Scaled_Ref.h5ad"
NEIGHBOR_PATH = "Vessel_Spatial_Neighbors_RE1.npy"
OUTPUT_FILE = "Final_Ensemble_Matching_Results_WT.csv"

def run_fast_ensemble_matching():
    print("[*] Starting Phase 2: High-Speed Realtime Ensemble Matching...")
    
    # 1. Load Data
    print("  - Loading Reference and ST Headers...")
    adata_ref = sc.read_h5ad(REF_PATH)
    adata_st = sc.read_10x_h5(ST_PATH)
    adata_st.var_names_make_unique()
    
    common_genes = adata_ref.var_names.intersection(adata_st.var_names)
    print(f"  - Aligned {len(common_genes)} genes.")
    
    adata_ref = adata_ref[:, common_genes]
    adata_st = adata_st[:, common_genes]
    
    st_raw = adata_st.X.tocsr()
    ref_raw = adata_ref.X.tocsr()
    num_spots = adata_st.shape[0]
    num_ref_cells = adata_ref.shape[0]
    
    # 2. Precompute Spatial Neighbors Mapping (Extremely Fast)
    print("  - Building fast neighbor mapping...")
    neighbors = np.load(NEIGHBOR_PATH)
    df_pos = pd.read_parquet(POS_PATH)
    df_pos = df_pos.reset_index(drop=True)
    
    bc_to_st_idx = {bc: i for i, bc in enumerate(adata_st.obs_names)}
    bc_to_pos_idx = {bc: i for i, bc in enumerate(df_pos['barcode'])}
    
    pos_idx_to_st_idx = np.full(len(df_pos), -1, dtype=np.int32)
    for i, bc in enumerate(df_pos['barcode']):
        if bc in bc_to_st_idx:
            pos_idx_to_st_idx[i] = bc_to_st_idx[bc]
            
    st_idx_to_pos_idx = np.array([bc_to_pos_idx[bc] for bc in adata_st.obs_names], dtype=np.int32)
    
    valid_neighbors_list = []
    for i in range(num_spots):
        pos_idx = st_idx_to_pos_idx[i]
        n_pos_idxs = neighbors[pos_idx]
        n_st_idxs = pos_idx_to_st_idx[n_pos_idxs]
        n_st_idxs = n_st_idxs[n_st_idxs != -1]
        valid_neighbors_list.append(n_st_idxs)
    
    # 3. GPU Matching Loop
    spot_batch_size = 1000
    ref_batch_size = 4000
    results = []
    
    print(f"[*] Executing 1:1 Matching with Realtime Spatial Pooling...")
    
    with torch.no_grad():
        for i in range(0, num_spots, spot_batch_size):
            start_batch = time.time()
            end_idx = min(i + spot_batch_size, num_spots)
            
            # --- REALTIME ENSEMBLE ---
            batch_dense = np.zeros((end_idx - i, len(common_genes)), dtype=np.float32)
            for local_i, st_idx in enumerate(range(i, end_idx)):
                n_idxs = valid_neighbors_list[st_idx]
                mask = n_idxs != st_idx
                
                # 1.0 * Self + 0.5 * Neighbors
                if mask.any():
                    neighbors_only = n_idxs[mask]
                    pooled = st_raw[st_idx].toarray() + 0.5 * st_raw[neighbors_only].sum(axis=0)
                else:
                    pooled = st_raw[st_idx].toarray()
                batch_dense[local_i] = pooled
                
            V = torch.tensor(batch_dense, dtype=torch.float32).to(DEVICE)
            V_denom = V.sum(dim=1, keepdim=True) + 1e-8 # Denominator is the pooled total reads
            
            best_pcts = torch.full((end_idx - i, 1), -1.0, dtype=torch.float32, device=DEVICE)
            best_ids = torch.zeros((end_idx - i, 1), dtype=torch.long, device=DEVICE)
            
            # --- CROSS REFERENCE ---
            for j in range(0, num_ref_cells, ref_batch_size):
                ref_end = min(j + ref_batch_size, num_ref_cells)
                S_batch = torch.tensor(ref_raw[j:ref_end].toarray(), dtype=torch.float32).to(DEVICE)
                
                # Matrix multiplication
                scores = torch.matmul(V, S_batch.t())
                pcts = (scores / V_denom) * 100
                
                max_pcts, max_idxs = torch.max(pcts, dim=1, keepdim=True)
                
                update_mask = max_pcts > best_pcts
                best_pcts[update_mask] = max_pcts[update_mask]
                best_ids[update_mask] = max_idxs[update_mask] + j
                
                del S_batch, scores, pcts
            
            # Collect Batch Results
            pcts_np = best_pcts.cpu().numpy().flatten()
            ids_np = best_ids.cpu().numpy().flatten()
            
            for k in range(end_idx - i):
                results.append({
                    'barcode': adata_st.obs_names[i + k],
                    'explanation_pct': pcts_np[k],
                    'best_cell_id': ids_np[k]
                })
            
            del V, V_denom, batch_dense, best_pcts, best_ids
            if i % 20000 == 0:
                print(f"    - [{i}/{num_spots}] spots ensemble-matched. Speed: {time.time() - start_batch:.2f}s/batch")
                gc.collect()

    # 4. Save Final CSV
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n[!] SUCCESS. Ensemble matching completed and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        run_fast_ensemble_matching()
    except Exception as e:
        print(f"[!] Critical Error: {e}")
    gc.collect()

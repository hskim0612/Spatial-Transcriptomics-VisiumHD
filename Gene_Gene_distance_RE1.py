#!/usr/bin/env python3
# Max RAM-efficient: Serpina3n distance to Gfap in RE-1 (2um) using h5py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import h5py
import os
import gc
from sklearn.neighbors import NearestNeighbors
import pyarrow.parquet as pq

# Paths
DATA_DIR = r"C:\Users\hskim\OneDrive - Vanderbilt\Lab-related\Results\20251117_Visium\CellRanger\13604-RE-1\binned_outputs\square_002um"
H5_PATH = os.path.join(DATA_DIR, "filtered_feature_bc_matrix.h5")
POS_PATH = os.path.join(DATA_DIR, "spatial", "tissue_positions.parquet")

print("=" * 50)
print("RE-1 Serpina3n to Gfap Distance Analysis (2um) - Ultra Low RAM")
print("=" * 50)

# 1. Load Spatial Coordinates
print("\n[1] Loading tissue positions...")
pos_df = pq.read_table(POS_PATH).to_pandas()
pos_df.set_index('barcode', inplace=True)

# 2. Extract Data directly from H5 without building full matrix
print("[2] Extracting Serpina3n and Gfap coordinates directly from H5...")
with h5py.File(H5_PATH, 'r') as f:
    # 10x Genomics H5 format stores strings as bytes, require decoding
    h5_barcodes = np.array([b.decode('utf-8') for b in f['matrix/barcodes'][:]])
    gene_names = np.array([g.decode('utf-8') for g in f['matrix/features/name'][:]])

    serp_idx = np.where(gene_names == 'Serpina3n')[0][0]
    gfap_idx = np.where(gene_names == 'Gfap')[0][0]

    # Load only the 1D arrays of the Compressed Sparse Column format
    indices = f['matrix/indices'][:]
    indptr = f['matrix/indptr'][:]
    data = f['matrix/data'][:]

def get_positive_coords(gene_idx):
    # Find exact locations in the data array where this gene is recorded
    p_locs = np.where(indices == gene_idx)[0]
    
    # Filter for expression > 0 
    p_locs = p_locs[data[p_locs] > 0]
    
    # Map back to column index (barcode index) using indptr
    col_indices = np.searchsorted(indptr, p_locs, side='right') - 1
    
    # Extract corresponding barcodes and coordinates
    target_barcodes = h5_barcodes[col_indices]
    coords = pos_df.loc[target_barcodes, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
    return coords

serp_coords = get_positive_coords(serp_idx)
gfap_coords = get_positive_coords(gfap_idx)

# Clear large arrays from RAM completely
del indices, indptr, data, h5_barcodes, pos_df, gene_names
gc.collect()

print(f"  Serpina3n positive spots: {len(serp_coords)}")
print(f"  Gfap positive spots: {len(gfap_coords)}")

# 3. Compute Distances
print("\n[3] Computing nearest neighbor distances...")
if len(gfap_coords) > 0 and len(serp_coords) > 0:
    tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gfap_coords)
    distances, _ = tree.kneighbors(serp_coords)
    dists = distances.flatten()
    
    print(f"  Mean distance: {np.mean(dists):.1f} px/um")
    print(f"  Median distance: {np.median(dists):.1f} px/um")
    print(f"  Standard deviation: {np.std(dists):.1f} px/um")
    print(f"  Minimum distance: {np.min(dists):.1f} px/um")
    print(f"  Maximum distance: {np.max(dists):.1f} px/um")
    
    zero_dist_count = np.sum(dists == 0)
    zero_dist_pct = (zero_dist_count / len(dists)) * 100
    print(f"\n  Co-expressed spots zero distance: {zero_dist_count} cases")
    print(f"  Co-expression ratio: {zero_dist_pct:.1f}%")
    
else:
    print("  Insufficient spots to calculate distance.")

print("\nDONE!")


"""
Pathology-Specific WM: Inflammation Inferno Index (Zero-Alpha Cutoff)
====================================================================
Logic:
1. Target 7 Genes: C3, Serpina3n, S100a4, Apoe, Spp1, Vim, Aif1l.
2. Equal Power (Z-score): Standardize each gene.
3. Zero-Alpha Mapping:
   - BACKGROUND: Fully Opaque (alpha=1.0).
   - INFERNO INDEX: 
     * Index <= 0 (Average or below) -> Fully Transparent (alpha=0.0).
     * Index > 0 -> Alpha increases linearly from 0.0 to 1.0 at Max value.
   - This ensures ONLY areas with positive inflammation signals are visible over the background.
4. Color: Blue (Cold) -> Yellow -> Red (Hot).

Created for Dr. Kim
Date: 2026-02-09
"""

import matplotlib
matplotlib.use('Agg')
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import gc
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class Config:
    root_path = Path(r"E:\20251117_Visium\CellRanger")
    output_dir = Path("Pathology_Specific_WM_Inflammation_Inferno_ZeroAlpha_Results")
    samples = {
        "WT": ["13604-RE-1", "13604-RE-4"],
        "5xFAD": ["13604-RE-2", "13604-RE-3"]
    }
    all_sample_list = ["13604-RE-1", "13604-RE-4", "13604-RE-2", "13604-RE-3"]
    inferno_genes = ['C3', 'Serpina3n', 'S100a4', 'Apoe', 'Spp1', 'Vim', 'Aif1l']
    wm_markers = ['Mbp', 'Plp1', 'Mog']
    gm_markers = ['Slc17a7', 'Neurod6', 'Gad1']

class InfernoVisualizer:
    def __init__(self, config):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_segment(self, sname):
        logger.info(f"  Processing: {sname}")
        data_path = self.config.root_path / sname / "binned_outputs" / "square_016um"
        h5_file = data_path / "filtered_feature_bc_matrix.h5"
        if not h5_file.exists(): h5_file = data_path / "raw_feature_bc_matrix.h5"
        
        try:
            adata = sc.read_10x_h5(str(h5_file))
            adata.var_names_make_unique()
            tp_parquet = data_path / "spatial" / "tissue_positions.parquet"
            coords = pd.read_parquet(tp_parquet) if tp_parquet.exists() else pd.read_csv(data_path / "spatial" / "tissue_positions.csv", index_col=0)
            if 'barcode' in coords.columns: coords = coords.set_index('barcode')
            common = adata.obs_names.intersection(coords.index)
            adata = adata[common].copy()
            spatial_coords = coords.loc[common, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values.astype(float)
            spatial_coords[:, 1] *= -1
            adata.obsm['spatial'] = spatial_coords
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Regional Segmentation
            valid = [g for g in (self.config.wm_markers + self.config.gm_markers) if g in adata.var_names]
            X = StandardScaler().fit_transform(adata[:, valid].X.toarray())
            pca = PCA(n_components=1, random_state=42)
            scores = pca.fit_transform(X).flatten()
            if pca.components_[0, valid.index('Mbp')] < 0: scores = -scores
            gmm = GaussianMixture(n_components=2, random_state=42).fit(scores.reshape(-1, 1))
            initial = np.where(gmm.predict(scores.reshape(-1, 1)) == np.argmax(gmm.means_), 'WM', 'GM')
            knn = KNeighborsClassifier(n_neighbors=15).fit(adata.obsm['spatial'], initial)
            adata.obs['Region_Type'] = knn.predict(adata.obsm['spatial'])
            return adata
        except Exception as e:
            logger.error(f"    Error loading {sname}: {e}")
            return None

    def calculate_and_visualize(self):
        logger.info(f">>> Calculating Zero-Alpha Inferno Index for 7 Genes...")
        sample_results = {}
        all_index_values = []

        for sname in self.config.all_sample_list:
            adata = self._load_and_segment(sname)
            if adata is None: continue
            spatial = adata.obsm['spatial'].copy()
            regions = adata.obs['Region_Type'].values
            z_scores = []
            for gene in self.config.inferno_genes:
                if gene in adata.var_names:
                    expr = adata[:, gene].X.toarray().flatten()
                    z = (expr - np.mean(expr)) / (np.std(expr) + 1e-9)
                    z_scores.append(z)
            
            if not z_scores: continue
            inferno_index = np.sum(z_scores, axis=0)
            sample_results[sname] = {'spatial': spatial, 'regions': regions, 'index': inferno_index}
            all_index_values.extend(inferno_index)
            del adata; gc.collect()

        if not sample_results: return

        vmax = np.percentile(all_index_values, 99)
        vmin = np.percentile(all_index_values, 1)

        fig, axes = plt.subplots(1, 4, figsize=(28, 7), facecolor='white')
        fig.suptitle("Zero-Alpha Inflammation Inferno Index (7-Gene Merger)", fontsize=26, fontweight='bold', y=1.05)
        
        colors = ["#0000ff", "#ffff00", "#ff0000"] 
        custom_cmap = LinearSegmentedColormap.from_list("inferno_rich", colors)

        for i, sname in enumerate(self.config.all_sample_list):
            ax = axes[i]
            if sname in sample_results:
                sr = sample_results[sname]
                spatial = sr['spatial']
                regions = sr['regions']
                idx = sr['index']
                
                # --- OPAQUE BACKGROUND ---
                ax.scatter(spatial[regions=='GM', 0], spatial[regions=='GM', 1], c='#7f7f7f', s=1.0, alpha=1.0) 
                ax.scatter(spatial[regions=='WM', 0], spatial[regions=='WM', 1], c='#d3d3d3', s=1.0, alpha=1.0) 
                
                # --- ZERO-ALPHA MAPPING ---
                # Alpha is 0 for idx <= 0, and scales to 1.0 at vmax
                alpha_vals = np.clip(idx / vmax, 0, 1)
                
                # Color mapping
                norm_idx = (idx - vmin) / (vmax - vmin)
                norm_idx = np.clip(norm_idx, 0, 1)
                rgba_colors = custom_cmap(norm_idx)
                
                # Apply Dynamic Alpha
                rgba_colors[:, 3] = alpha_vals
                
                # Sort to draw high-alpha (intense) points on top
                sort_idx = np.argsort(idx)
                ax.scatter(spatial[sort_idx, 0], spatial[sort_idx, 1], 
                           c=rgba_colors[sort_idx], s=1.2)
                
                if i == 3:
                    scat = ax.scatter(spatial[0,0], spatial[0,1], c=idx[0], 
                                     cmap=custom_cmap, vmin=vmin, vmax=vmax, s=0)
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
                    fig.colorbar(scat, cax=cbar_ax, label='Inferno Index (Z-Sum)')
            
            cond = "WT" if "RE-1" in sname or "RE-4" in sname else "5xFAD"
            ax.set_title(f"{sname}\n({cond})", fontsize=20, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 0.91, 0.95])
        out_file = self.config.output_dir / "ZeroAlpha_Inferno_Index_Comparison.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f">>> FINAL RESULT GENERATED: {out_file}")

if __name__ == "__main__":
    conf = Config()
    viz = InfernoVisualizer(conf)
    viz.calculate_and_visualize()

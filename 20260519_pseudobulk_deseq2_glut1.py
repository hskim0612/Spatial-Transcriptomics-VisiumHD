"""
Memory-Efficient Pseudo-bulk Analysis for Spatial Transcriptomics
WT vs 5xFAD Vessel GLUT1 Comparison using DESeq2 Size Factors
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==============================================================================
# 1. 샘플별 독립적 로딩 및 혈관 spot 선별 함수
# ==============================================================================
def process_sample_to_pseudobulk(base_path, sample_name):
    """
    개별 샘플을 로드하고 sparse matrix 상태에서 혈관을 선별한 뒤
    해당 혈관 spot들의 raw count를 합산하여 pseudo-bulk 벡터로 반환합니다.
    """
    print(f"[{sample_name}] 데이터 로딩 및 분석 중...")
    data_path = os.path.join(base_path, sample_name, "binned_outputs", "square_016um")
    h5_file = os.path.join(data_path, "filtered_feature_bc_matrix.h5")
    
    # Sparse matrix 상태로 로드
    adata = sc.read_10x_h5(h5_file)
    adata.var_names_make_unique()
    
    # 좌표 데이터 로딩 및 필터링
    parquet_file = os.path.join(data_path, "spatial", "tissue_positions.parquet")
    coords = pd.read_parquet(parquet_file)
    if 'barcode' in coords.columns:
        coords = coords.set_index('barcode')
        
    common = adata.obs_names.intersection(coords.index)
    adata = adata[common, :].copy()
    
    # Raw count 보존
    adata.layers['counts'] = adata.X.copy()
    
    # 혈관 선별을 위한 임시 정규화 진행
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 혈관 마커 스코어링
    vessel_markers = ['Cldn5', 'Flt1', 'Pecam1', 'Slco1c1', 'Vwf', 'Abcb1a']
    valid_genes = [g for g in vessel_markers if g in adata.var_names]
    sc.tl.score_genes(adata, valid_genes, score_name='Vessel_Score')
    
    # 개별 샘플 단위 Gaussian Mixture Model 적용
    vessel_scores = adata.obs['Vessel_Score'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(vessel_scores)
    
    # Score가 더 높은 군집을 혈관으로 정의
    label_mean_0 = vessel_scores[labels == 0].mean()
    label_mean_1 = vessel_scores[labels == 1].mean()
    vessel_cluster = 0 if label_mean_0 > label_mean_1 else 1
    adata.obs['Is_Vessel'] = (labels == vessel_cluster)
    
    # 혈관 spot 선별 및 Pseudo-bulk 압축
    adata_vessel = adata[adata.obs['Is_Vessel']]
    num_vessel_spots = adata_vessel.n_obs
    print(f"  -> 식별된 혈관 spot 수: {num_vessel_spots}")
    
    # Raw counts를 sum하여 단일 벡터 생성
    pseudo_bulk_counts = np.array(adata_vessel.layers['counts'].sum(axis=0)).flatten()
    
    return pd.Series(pseudo_bulk_counts, index=adata.var_names), num_vessel_spots

# ==============================================================================
# 2. 메인 실행 및 Pseudo-bulk 통합
# ==============================================================================
ROOT_PATH = r"E:/20251117_Visium/CellRanger"
SAMPLE_LIST = ["13604-RE-1", "13604-RE-2", "13604-RE-3", "13604-RE-4"]
CONDITIONS = {
    "13604-RE-1": "WT", "13604-RE-4": "WT",
    "13604-RE-2": "5xFAD", "13604-RE-3": "5xFAD"
}

OUTPUT_DIR = "20260519_160600_pseudobulk_deseq2_glut1_vessel_wt_5xfad_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("개별 GMM 및 Pseudo-bulk 생성 시작")
print("=" * 70)

bulk_data = {}
spot_counts = {}

for sample in SAMPLE_LIST:
    bulk_series, n_spots = process_sample_to_pseudobulk(ROOT_PATH, sample)
    bulk_data[sample] = bulk_series
    spot_counts[sample] = n_spots

# DataFrame 구축 (행: 샘플, 열: 유전자)
count_matrix = pd.DataFrame(bulk_data).T
count_matrix.fillna(0, inplace=True)
count_matrix = count_matrix.astype(int)

# 메타데이터 구축
metadata = pd.DataFrame({
    'Condition': [CONDITIONS[s] for s in SAMPLE_LIST],
    'Vessel_Spots': [spot_counts[s] for s in SAMPLE_LIST]
}, index=SAMPLE_LIST)

print("\nPseudo-bulk 카운트 매트릭스 차원:", count_matrix.shape)

# ==============================================================================
# 3. DESeq2 기반 통계 분석 및 Size Factor 정규화
# ==============================================================================
print("\n" + "=" * 70)
print("PyDESeq2를 활용한 정규화 및 통계 분석")
print("=" * 70)

# PyDESeq2 데이터셋 생성
dds = DeseqDataSet(
    counts=count_matrix,
    metadata=metadata,
    design_factors="Condition"
)

# Size factor 계산 및 통계 모델 피팅
dds.deseq2()

# 정규화된 데이터 추출
normalized_counts = dds.layers["normed_counts"]
normalized_df = pd.DataFrame(normalized_counts, index=count_matrix.index, columns=count_matrix.columns)

# WT를 기준 그룹으로 설정하여 통계 검정 수행
stat_res = DeseqStats(dds, contrast=["Condition", "5xFAD", "WT"])
stat_res.summary()
results_df = stat_res.results_df

# 혈관 마커 및 GLUT1 추출
target_genes = ['Slc2a1', 'Cldn5', 'Flt1', 'Pecam1', 'Slco1c1', 'Vwf', 'Abcb1a']
available_targets = [g for g in target_genes if g in normalized_df.columns]

# 결과 요약
print("\n[Target Genes DESeq2 Results]")
target_results = results_df.loc[available_targets, ['log2FoldChange', 'pvalue', 'padj']]
print(target_results)

# 저장
target_results.to_csv(os.path.join(OUTPUT_DIR, "Target_Genes_DESeq2_Stats.csv"))
normalized_df[available_targets].to_csv(os.path.join(OUTPUT_DIR, "Normalized_Target_Counts.csv"))

# ==============================================================================
# 4. 시각화 (GLUT1 발현량 비교)
# ==============================================================================
glut1_gene = 'Slc2a1'
if glut1_gene in normalized_df.columns:
    wt_expr = normalized_df.loc[metadata['Condition'] == 'WT', glut1_gene]
    ad_expr = normalized_df.loc[metadata['Condition'] == '5xFAD', glut1_gene]
    
    pval = target_results.loc[glut1_gene, 'pvalue']
    lfc = target_results.loc[glut1_gene, 'log2FoldChange']
    
    plt.figure(figsize=(6, 5))
    plt.bar(['WT', '5xFAD'], [wt_expr.mean(), ad_expr.mean()], 
            yerr=[wt_expr.std(), ad_expr.std()],
            capsize=5, color=['lightblue', 'salmon'], alpha=0.8, edgecolor='black')
    
    # 개별 샘플 산점도 추가
    plt.scatter(np.zeros(len(wt_expr)), wt_expr, color='blue', zorder=3, label='WT Samples')
    plt.scatter(np.ones(len(ad_expr)), ad_expr, color='red', zorder=3, label='5xFAD Samples')
    
    plt.title(f'GLUT1 (Slc2a1) Pseudo-bulk Expression\nLog2FC = {lfc:.2f}, p-value = {pval:.2e}', fontweight='bold')
    plt.ylabel('DESeq2 Normalized Expression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "GLUT1_Pseudobulk_Comparison.png"), dpi=300)
    plt.close()
    print("\n  -> 시각화 결과 저장 완료: GLUT1_Pseudobulk_Comparison.png")

#Bulk RNA-seq based on RandomForest
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
import re
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance


data_path = 'C:/Center_bioinfo/ML_bulk_RNA_seq/'
cts_name = 'GSE132040_counts.csv'
meta_name = 'GSE132040_metadata.csv'

counts_df = pd.read_csv(data_path + cts_name, index_col=0)
meta_df = pd.read_csv(data_path + meta_name)

name_pattern = r'\.gencode.+'
useless_gene_remove_pattern = r'^__.'

na_sum = {}
for i in meta_df.columns:
    na_sum[i] = meta_df[i].isna().sum()

remove_index = counts_df.index.str.match(useless_gene_remove_pattern)
useful_counts_df = counts_df.loc[~remove_index]
useful_counts_df_trans = useful_counts_df.T

useful_counts_df_trans.index = useful_counts_df_trans.index.str.replace(name_pattern, "", regex=True)

meta_df['sample'] = meta_df['Sample name'].astype(str).str.strip()

meta_sample = pd.Index(meta_df['sample'].sort_values())

na_pattern = meta_df['characteristics: age'].str.match(r'.*NA.+$')

meta_df_nw = meta_df[~na_pattern].copy()

valid = meta_df_nw['sample']
align_to_count = valid[valid.isin(useful_counts_df_trans.index)]

counts_final = useful_counts_df_trans.loc[align_to_count]

lib_size = counts_final.sum(axis=1)

cpm_counts = counts_final.divide(lib_size, axis=0) * 1e6

threshold_keepgene = (cpm_counts >= 1).sum(axis=0) >= cpm_counts.shape[0] * 0.3

counts_final_keeped = counts_final.loc[:, threshold_keepgene]

meta_df_aligned = (
    meta_df_nw
    .set_index('sample')
    .reindex(counts_final_keeped.index.astype(str).str.strip())
)

meta_df_aligned['age'] = meta_df_aligned['characteristics: age'].astype(int)
dds = DeseqDataSet(counts=counts_final_keeped, metadata=meta_df_aligned, design="~1", n_cpus=4, size_factors_fit_type='poscounts'\
                  , quiet=False)

dds.fit_size_factors()
dds.vst(use_design=False)

vst_mat = dds.layers['vst_counts']
vst_df = pd.DataFrame(vst_mat, index=dds.obs_names, columns=dds.var_names)

need_cols = ['age']
meta_aln = meta_df_aligned.loc[vst_df.index, [c for c in need_cols]].copy()
y_age = meta_aln['age'].astype(float).values

pca_test = PCA(random_state=40, n_components=10)
pca_run = pca_test.fit_transform(vst_df.values)
expl = pca_test.explained_variance_ratio_
print([f"Top 5 axis explained variance: {v}" for v in expl])

fig= plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter3D(pca_run[:, 0], pca_run[:, 1], pca_run[:, 2], s=16, c=y_age, cmap='viridis')
ax.set_xlabel(f'PC1: {expl[0]:.2%}'); ax.set_ylabel(f'PC2: {expl[1]:.2%}'); ax.set_zlabel(f'PC3: {expl[2]:.2%}')
cb = plt.colorbar(sc); cb.set_label('age')
ax.set_title('PCA on VST')
plt.tight_layout()
plt.savefig('pca_vst', dpi=300, format='png', bbox_inches='tight')
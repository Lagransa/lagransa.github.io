#Bulk RNA-seq based on RandomForest
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from scipy.stats import spearmanr
import shap
from shap import TreeExplainer
import re
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance


data_path = '/parastor/home/dongpc/git_repo/lagransa.github.io/bulk_rna_seq_ml/dataset/'
# data_path = 'C:/Center_bioinfo/ML_bulk_RNA_seq/'
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
dds = DeseqDataSet(counts=counts_final_keeped, metadata=meta_df_aligned, design="~1", n_cpus=-1, size_factors_fit_type='poscounts'\
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

#整理数据
X_df, gene = vst_df, vst_df.columns.to_numpy()
y = meta_df_aligned['age']
mask = np.argsort(-X_df.values.var(axis=0)).astype(int)[:1000]
vst_df_o = vst_df.reset_index().iloc[:, 1:]
vst_df_vared = vst_df_o.iloc[:, mask]
vst_df_vared['Acta1'].values[:10]

#分层抽样留盲测集
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
train_idx, test_idx = next(sss.split(vst_df_vared.values, y))
X_train, X_test = vst_df_vared.iloc[train_idx], vst_df_vared.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


#随机森林
class RF():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth':[None, 5, 15, 30],
            'max_features':['sqrt', 0.1, 0.2, 0.5],
            'min_samples_leaf': [1, 2, 3, 5],
            'min_samples_split': [2, 3, 4]
        }
        self.model = None
    
    #模型调优
    def forward(self):
        score_acc = ['accuracy', 'balanced_accuracy', 'f1_macro', 'neg_log_loss']
        best_model = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                           param_grid=self.param_grid,
                           scoring=score_acc,
                           cv=5,
                           n_jobs=-1)
        best_model.fit(self.X, self.y)
        best_param = best_model.best_params_
        best_score = best_model.best_score_
        best_model_acc = best_model['test_accuracy'].mean()
        best_model_bacc = best_model['test_balanced_accuracy'].mean()
        best_model_macro = best_model['test_f1_macro'].mean()
        best_model_neg_log_loss = -best_model['test_neg_log_loss'].mean()
        print(f'模型最佳参数为:{best_param}, 最佳模型参数得分为{best_score}, ACC为{best_model_acc}, bACC为{best_model_bacc}, MACRO为{best_model_macro}, 负对数损失为{best_model_neg_log_loss}')
        y_pred = best_model.predict(self.X)
        train_acc = classification_report(y_pred, self.y)
        print(train_acc)
        with open('train_acc.txt', 'w+') as f:
            f.writelines(train_acc)
        self.model = best_model.best_estimator_

    def prediction(self):
        y_pred = self.model.predict(self.X_test)
        final_acc = classification_report(self.y_test, y_pred)
        print(final_acc)
        with open('RF_test_acc.txt', 'w+') as f:
            f.writelines(final_acc)

    def feature_importance_val(self, n_repeats=30, top_k=10):
        #gini重要度
        importance = self.model.feature_importances_
        feature_names = self.X.columns
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
            }).sort_values('Importance', ascending=False)
        imp_x = feature_df['Importance'].values[:top_k]
        imp_y = feature_df['Feature'].values[:top_k]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=imp_x, y=imp_y)
        plt.title('Gini Importance Ranking by RF')
        plt.savefig(f'RF_gini_importance_ranking_top{top_k}.png', bbox_inches='tight', dpi=300)
        feature_df.to_csv('RF_gini_importance_ranking.csv', index=False)
        plt.close()
        
        #置换重要度
        permutation_cal = permutation_importance(
            estimator=self.model,
            X=self.X_test,
            y=self.y_test,
            n_repeats=n_repeats,
            n_jobs=-1,
            random_state=7,
            scoring='accuracy'
        )
        
        per_importance = pd.DataFrame({
            'Feature': feature_names,
            'Permutation_importance_mean': permutation_cal.importances_mean,
            'Permutation_importance_std': permutation_cal.importances_std
                                      }).sort_values('Permutation_importance_mean', ascending=False)
        per_x = per_importance['Permutation_importance_mean'].values[: top_k]
        per_y = per_importance['Feature'].values[: top_k]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=per_x, y=per_y)
        plt.title('Permutation importance Ranking by RF')
        plt.savefig(f'RF_permutation_importance_ranking_top{top_k}.png', bbox_inches='tight', dpi=300)
        per_importance.to_csv('RF_permutation_importance_ranking.csv', index=False)
        plt.close()

        #SHAP特征重要度
        shap_explainer = TreeExplainer(self.model)
        shap_values = shap_explainer.shap_values(self.X_test)
        with open('RF_shap_values.txt', 'w+') as f:
            f.writelines(shap_values)
        shap.summary_plot(shap_values, self.X_test, feature_names=per_y, plot_type='bar')
        plt.savefig(f'RF_SHAP_ranking_top{top_k}.png', bbox_inches='tight', dpi=300)
        plt.close()

rf_model = RF(X_train, y_train, X_test, y_test) #实例化
rf_model.forward() #forward
rf_model.prediction() #预测分数
rf_model.feature_importance_val() #拿特征重要度排序

class RFE_RF():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_filtered = None
        self.X_test_filtered = None
        self.final_model = None

    def forward(self):
        score_acc = ['accuracy', 'balanced_accuracy', 'f1_macro', 'neg_log_loss']
        model = RandomForestClassifier(n_estimators=100, max_features=0.1, random_state=7, n_jobs=-1)
        rfecv = RFECV(estimator=model, step=0.05, cv=5, scoring=score_acc, min_features_to_select=50, verbose=1, n_jobs=-1)
        selector = rfecv.fit(self.X, self.y)
        best_model_acc = rfecv['test_accuracy'].mean()
        best_model_bacc = rfecv['test_balanced_accuracy'].mean()
        best_model_macro = rfecv['test_f1_macro'].mean()
        best_model_neg_log_loss = -rfecv['test_neg_log_loss'].mean()
        print(f'模型ACC为{best_model_acc}, bACC为{best_model_bacc}, MACRO为{best_model_macro}, 负对数损失为{best_model_neg_log_loss}')

        self.X_train_filtered = selector.transform(self.X)
        self.X_test_filtered = selector.transform(self.X_test)
        print(f'选择的特征掩码为:{selector.support_}', f'特征排名为{selector.ranking_}', f'经选择后训练集的shape为{self.X_train_filtered.shape}')

    def prediction(self):
        final_model = RandomForestClassifier(n_estimator=200, max_features='sqrt', random_state=7, n_jobs=-1, max_depth=20)
        final_model.fit(self.X_train_filtered, self.y)
        final_pred = final_model.predict(self.X_test_filtered)
        final_acc = classification_report(self.y_test, final_pred)
        with open('RFE_test_acc.txt', 'w+') as f:
            f.writelines(final_acc)

    def feature_importance_val(self, n_repeats=30, top_k=10):
        importance_gini = self.final_model.feature_importances_
        feature_names = self.X_train_filtered.columns
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_gini
        }).sort_values('Importance', ascending=False)
        imp_x = feature_df['Importance'].values[:top_k]
        imp_y = feature_df['Feature'].values[:top_k]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=imp_x, y=imp_y)
        plt.title('Gini Importance Ranking by RFE')
        plt.savefig(f'RFE_gini_importance_ranking_top{top_k}.png', bbox_inches='tight', dpi=300)
        plt.close()
        feature_df.to_csv('RFE_gini_importance_ranking.csv', index=False)
        
        permutation_cal = permutation_importance(estimator=self.final_model,
                                                 X=self.X_test_filtered,
                                                 y=self.y_test,
                                                 scoring='accuracy',
                                                 n_jobs=-1,
                                                 n_repeats=n_repeats,
                                                 random_state=7
                                                )
        per_importance = pd.DataFrame({
            'Feature': self.X_test_filtered.columns,
            'Permutation Importance mean': permutation_cal.importances_mean,
            'Permutation Importance std': permutation_cal.importances_std
        }).sort_values('Permutation Importance mean', ascending=False)
        per_x = per_importance['Permutation Importance mean'].values[:top_k]
        per_y = per_importance['Feature'].values[:top_k]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=per_x, y=per_y)
        plt.title('Permutation Importance Ranking by RFE')
        plt.savefig(f'RFE_permutation_importance_ranking_top{top_k}.png', bbox_inches='tight', dpi=300)
        plt.close()
        per_importance.to_csv('RFE_permutation_importance_ranking.csv', index=False)

        #SHAP特征重要度
        shap_explainer = TreeExplainer(self.final_model)
        shap_values = shap_explainer.shap_values(self.X_test_filtered)
        with open('RFE_shap_values.txt', 'w+') as f:
            f.writelines(shap_values)
        shap.summary_plot(shap_values, self.X_test_filtered, feature_names=per_y, plot_type='bar')
        plt.savefig(f'RFE_SHAP_ranking_top{top_k}.png', bbox_inches='tight', dpi=300)
        plt.close()

        
    
rfe = RFE_RF()
rfe.forward()
rfe.prediction()
rfe.feature_importance_val()
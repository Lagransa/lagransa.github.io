RData_path <- 'your_RData_path.RData'
RData_save <- 'your_newRData.RData'


load(file=RData_path)

smallest_UMI <- 500
smallest_Genes <- 250
log10GenesPerUMI_threshold <- 0.8
mito_Ratio_threshold <- 0.2
no_zero_genes_threshold <- 10


#过滤
filtered_seurat <- subset(x=raw_seurat, subset=(nUMI>smallest_UMI)&
                          (nGene>=smallest_Genes)&
                         (log10GenesPerUMI>log10GenesPerUMI_threshold)&
                         (mito_Ratio<mito_Ratio_threshold))

###多样本多做一步
filtered_seurat <- JoinLayers(filtered_seurat)

#过滤后新对象
counts <- GetAssayData(object=filtered_seurat, layer='counts')
nozero <- counts>0
keep_genes <- Matrix::rowSums(nozero)>=no_zero_genes_threshold
filtered_counts <- counts[keep_genes,]
filtered_seurat <- CreateSeuratObject(filtered_counts, meta.data=filtered_seurat@meta.data)

save(filtered_seurat, file=RData_save)
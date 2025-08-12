RData_path <- 'your_RData_path.RData'
RData_save <- 'your_newRData.RData'


load(file=RData_path)

#####################extra:细胞周期数据
cycle_URL <- 'https://raw.githubusercontent.com/hbc/tinyatlas/master/cell_cycle/物种名.csv'
cc_file <- getURL(cycle_URL)
cell_cycle_genes <- read.csv(text=cc_file)
ah <- AnnotationHub()
ahDb <- query(ah, pattern=c('种名', 'EnsDb'), ignore.case=TRUE)
id <- ahDb %>% mcols() %>% rownames() %>% tail(n=1)
edb <- ah[[id]]
annotations <- genes(edb, return.type='data.frame')
annotations <- annotations %>% dplyr::select(gene_id, gene_name, seq_name, gene_biotype, description)
cell_cycle_markers <- dplyr::left_join(cell_cycle_genes, annotations, by=c('geneID'='gene_id'))

s_genes <- cell_cycle_markers %>% dplyr::filter(phase=='S') %>% pull('gene_name')
g2m_genes <- cell_cycle_markers %>% dplyr::filter(phase=='G2M') %>% pull('gene_name')


#单样本正则+PCA
seurat_phase <- NormalizeData(filtered_seurat)
seurat_phase <- CellCycleScoring(seurat_phase, g2m.features=g2m_genes, s.features=s_genes)
seurat_phase <- FindVariableFeatures(seurat_phase, selection.method='vst', nfeatures=2000, verbose=FALSE)
seurat_phase <- ScaleDaita(seurat_phase)
seurat_integrated <- RunPCA(seurat_phase)

###多样本正则
#########改最大Object限制
options(future.globals.maxSize=4000*1024^2)

split_seurat <- SplitObject(filtered_seurat, split.by='sample')
split_seurat <- split_seurat[c(project_name[1], project_name[2])]
for( i in 1:length(split_seurat)){
    split_seurat[[i]] <- NormalizeData(split_seurat[[i]], verbose=TRUE)
    split_seurat[[i]] <- CellCycleScoring(split_seurat[[i]], g2m.features=g2m_genes, s.features=s_genes)
    split_seurat[[i]] <- SCTransform(split_seurat[[i]], vars.to.regress=c('mito_Ratio'))
    }
########CCA样本锚，只适用于多样本
integ_features <- SelectIntegrationFeatures(object.list=split_seurat, nfeatures=3000)
split_seurat <- PrepSCTIntegration(object.list=split_seurat, anchor.features=integ_features)

integ_anchors <- FindIntegrationAnchors(object.list=split_seurat, normalization.method='SCT', anchor.features=integ_features)
seurat_integrated <- IntegrateData(anchorset=integ_anchors, normalization.method='SCT')

save(seurat_integrated, file=RData_save)
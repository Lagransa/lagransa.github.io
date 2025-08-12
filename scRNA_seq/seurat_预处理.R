# args <- commandArgs(trailingOnly=TRUE)

# num_samples = length(args)
# args[1] = data_path
# args[2:] = sample_lst


# if (num_samples=2){
#     raw<count <- Read10X(data.dir=data_path)
# }

data_path <- 'input_your_folder'
min_features <- 100
project_name <- c('sample1')
save_path <- 'input_your_RDataStorage_folder.RData'

#single_sample
raw_count <- Read10X(data.dir=data_path)
raw_seurat <- CreateSeuratObject(counts=raw_count, min.features=min_features, project=project_name[1])



#multi_samples such as 2 samples
for(i in project_name){
    raw_count <- Read10X(data.dir=paste0(data_path, i))
    raw_object <- CreateSeuratObject(counts=raw_count, min.features=min_features, project=i)
    assign(i, raw_object)
}
raw_seurat <-  merge(x=project_name[1], y=project_name[2], add.cell.id=c('1', '2'))




#各种百分比
species_pattern <- '^mt-'

raw_seurat$log10GenesPerUMI <- log10(raw_seurat$nFeature_RNA) / log10(raw_seurat$nCount_RNA)
raw_seurat$mito_Ratio <- PercentageFeatureSet(object=raw_seurat, pattern=species_pattern)
raw_seurat$mito_Ratio <- raw_seurat$mito_Ratio / 100

#元数据框修改
meta_data <- raw_seurat@meta.data
meta_data$cells <- rownames(meta_data)
meta_data <- meta_data %>% dplyr::rename(data_folder=orig.ident, nUMI=nCount_RNA, nGene=nFeature_RNA)
####多样本数据框附加
meta_data$sample <- NA
meta_data$sample[which(str_detect(meta_data$cells, '^1_'))] <- '1'
meta_data$sample[which(str_detect(meta_data$cells, '^2_'))] <- '2'


raw_seurat@meta.data <- meta_data
save(raw_seurat, file=save_path)
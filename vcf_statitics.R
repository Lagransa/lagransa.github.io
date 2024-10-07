library(ggplot2)
library(tidyverse)

# args[1] = prefix, args[2].... = extension of these statistic files

args <- commandArgs(trailingOnly=TRUE)


alle_freq <- paste(args[1], '.', args[2], sep='')
indv_mean_depth <- paste(args[1], '.', args[3], sep='')
site_mean_depth <- paste(args[1], '.', args[4], sep='')
site_quality <- paste(args[1], '.', args[5], sep='')
missing_data_per_indv <- paste(args[1], '.', args[6], sep='')
missing_data_per_site <- paste(args[1], '.', args[7], sep='')
heterozygosity <- paste(args[1], '.', args[8], sep='')

draw_lst <- c(site_quality, site_mean_depth, indv_mean_depth, missing_data_per_indv, missing_data_per_site, alle_freq, heterozygosity)

#Site quality
qual_data <- read_delim(draw_lst[1], col_names=c('chrom', 'pos', 'qual'), skip=1)
p1 <- ggplot(qual_data, aes(qual)) + geom_density(fill='dodgerblue1', colour='blue', alpha=0.3)
p1 <- p1 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20)) + xlim(0, 10000)
ggsave('site_qual.pdf', p1, scale=3, dpi=300)

#Site mean depth
sdepth_data <- read_delim(draw_lst[2], col_names=c("chr", "pos", "mean_depth", "var_depth"), skip=1)
p2 <- ggplot(sdepth_data, aes(mean_depth)) + geom_density(fill='dodgerblue1', colour='blue', alpha=0.3)
p2 <- p2 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20)) + xlim(0, 50)
ggsave('site_depth.pdf', p2, scale=3, dpi=300)

#indv mean depth
idepth_data <- read_delim(draw_lst[3], col_names=c("ind", "nsites", "depth"), skip=1)
p3 <- ggplot(idepth_data, aes(depth)) + geom_histogram(fill='dodgerblue1', colour='blue', alpha=0.3)
p3 <- p3 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('individual_depth.pdf', p3, scale=3, dpi=300)

#missing data per indv
imiss_data <- read_delim(draw_lst[4], col_names=c("ind", "ndata", "nfiltered", "nmiss", "fmiss"), skip=1)
p4 <- ggplot(imiss_data, aes(fmiss)) + geom_histogram(fill='dodgerblue1', colour='blue', alpha=0.3)
p4 <- p4 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('individual_missing_data.pdf', p4, scale=3, dpi=300)

#missing data per site
lmiss_data <- read_delim(draw_lst[5], col_names=c("ind", "ndata", "nfiltered", "nmiss", "fmiss"), skip=1)
p5 <- ggplot(lmiss_data, aes(fmiss)) + geom_histogram(fill='dodgerblue1', colour='blue', alpha=0.3)
p5 <- p5 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('site_missing_data.pdf', p5, scale=3, dpi=300)

freq_data <- read_delim(draw_lst[6], delim='\t', col_names=c('chrom', 'pos', 'n_alleles', 'chr', 'a1', 'a2'), skip=1)
freq_data$min_allele <- freq_data %>% select(a1, a2) %>% apply(1, min)
p6 <- ggplot(freq_data, aes(min_allele)) + geom_density(fill='dodgerblue1', colour='blue', alpha=0.3)
p6 <- p6 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('allele_freq.pdf', p6, scale=3, dpi=300)

#heterozygosity
het_data <- read_delim(draw_lst[7], col_names=c("ind","ho", "he", "nsites", "f"), skip=1)
p7 <- ggplot(het_data, aes(f)) + geom_histogram(fill='dodgerblue1', colour='blue', alpha=0.3)
p7 <- p7 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('heterozygosity.pdf', p7, scale=3, dpi=300)
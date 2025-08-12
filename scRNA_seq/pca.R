library(ggplot2)
library(scatterplot3d)
library(tidyverse)

#input_prefix eigenvalue eigenvector
args <- commandArgs(trailingOnly=TRUE)

prefix <- args[1]
evalue <- paste(prefix, '.', 'eigenval', sep='')
evector <- paste(prefix, '.', 'eigenvec', sep='')


eigenval <- scan(evalue)
eigenvec <- read_delim(evector, colnames=FALSE)
eigenvec <- eigenvec[, -1]
num_col <- ncol(eigenvec)
names(eigenvec)[1] <- 'ind'
names(eigenvec)[2: num_col] <- paste('PC', 1: (num_col - 1))

#species & location if have
# spp <- rep('NA', length(eigenvec$ind))
# loc <- rep('NA', length(eigenvec$ind))
# spp[grep('spec1_name', eigenvec$ind)] <- 'spe1'
# spp[grep('spec2_name', eigenvec$ind)] <- 'spe2'
# loc[grep('loc1_name', eigenvec$ind)] <- 'location1'
# loc[grep('loc2_name', eigenvec$ind)] <- 'location2'

# pca <- as.tibble(data.frame(eigenvec, spp, loc))

#percentage of eigenvalue
pct <- data.frame(PC = 1: (num_col-1), pve = eigenval/sum(eigenval)*100)
p1 <- ggplot(pct, aes(PC, pve)) + geom_bar(stat='identity') + ylab('Percentage variance explained')
p1 <- p1 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('percentage_of_variance.pdf', p1, scale=3, dpi=300)


# do not use coord_equal if the value of x and y have huge difference
p2 <- ggplot(eigenvec, aes(PC1, PC2)) + geom_point(size = 3)
p2 <- p2 + scale_colour_manual(values=c('red'))
p2 <- p2 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('PC1_PC2.pdf', p2, scale=3, dpi=300)

p3 <- ggplot(eigenvec, aes(PC1, PC3)) + geom_point(size = 3)
p3 <- p3 + scale_colour_manual(values=c('red'))
p3 <- p3 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('PC1_PC3.pdf', p3, scale=3, dpi=300)

p4 <- ggplot(eigenvec, aes(PC2, PC3)) + geom_point(size = 3)
p4 <- p4 + scale_colour_manual(values=c('red'))
p4 <- p4 + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), panel.background=element_rect(fill='transparent'), axis.line=element_line(colour='black'), axis.text=element_text(size=14), axis.title=element_text(size=20))
ggsave('PC2_PC3.pdf', p4, scale=3, dpi=300)

#assume there are 3 levels
graphics.off()
pdf('PC_1_2_3.pdf', width=10, height=10)
# scatterplot3d(eigenvec[, 2:4], pch=c(16, 17, 18), col=c('red','steelblue','green'))
scatterplot3d(eigenvec[, 2:4], pch=16, color='steelblue')
dev.off()

library('caret')
library('plyr')
library('dplyr')
library('mlbench')
library('ggplot2')
library('plotly')
library('psych')
library('repr')
library('reshape2')
library('scatterplot3d')


# par()              # view current settings
opar <- par()      # make a copy of current settings
# par(opar)          # restore original settings  

set.seed(1)

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
    cormat[upper.tri(cormat)] <- NA
    return(cormat)
}

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
}

reorder_cormat <- function(cormat){
    # Use correlation between variables as distance
    dd <- as.dist((1-cormat)/2)
    hc <- hclust(dd)
    cormat <-cormat[hc$order, hc$order]
}

path <- '../data/hepatitisC.csv'
hepc.backup <- read.csv(path)
hepc <- read.csv(path)

# dimension of data
dim(hepc)

# view the first 5 rows of data
head(hepc)

# view the last 5 rows of data
tail(hepc)

summary(hepc$Category)

hepc.summary <- data.frame(table(hepc$Category))
names(hepc.summary) <- c('Category', 'Frequency')

ggplot(data=hepc.summary, aes(x=Category, y=Frequency)) +
        geom_bar(stat="identity", fill="steelblue")+
   geom_text(aes(label=Frequency), vjust=-0.3, size=3.5, alpha=1) + theme_minimal()

# filter the suspect blood donor
mask <- hepc$Category != 'suspect Blood Donor'
hepc <- hepc[mask,]

summary(hepc$Category)

# change the category
mask <- hepc$Category != 'Blood Donor'
hepc[mask, 'general_category'] <- 'Hepatitis'
hepc[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
hepc$Category <- factor(hepc$Category)
hepc$general_category <- factor(hepc$general_category)

summary(hepc$Category)
summary(hepc$general_category)

hepc.summary <- data.frame(table(hepc$general_category))
names(hepc.summary) <- c('general_category', 'Frequency')

ggplot(data=hepc.summary, aes(x=general_category, y=Frequency)) +
        geom_bar(stat="identity", fill="steelblue")+
   geom_text(aes(label=Frequency), vjust=-0.3, size=3.5, alpha=1) + theme_minimal()

# to check how many missing values is there
table(is.na(hepc))

# to check where the value located
apply(is.na(hepc), 2, which)

# taking missing columns
missing_cols <- colnames(hepc)[colSums(is.na(hepc)) > 0]
missing_cols

# preview before we impute missing values with median
summary(hepc[, missing_cols])

# impute missing value with median for each columns by category
for (col in missing_cols){
    hepc[, col] <- ave(hepc[, col], 
                       hepc$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# check on the is there any difference
summary(hepc[, missing_cols])

# double check if value is imputed
table(is.na(hepc))

numeric_cols <- colnames(hepc)[lapply(hepc, is.numeric) > 0]
numeric_cols

selected_cols <- numeric_cols[-1]
selected_cols

summary(hepc[, numeric_cols])

hepc.sc <- scale(hepc[, numeric_cols])

options(repr.plot.width=10, repr.plot.height=7) 
boxplot(hepc[, numeric_cols])

# melt the dataframe into suitable shape for plotting
hepc.mC <- melt(hepc[, c(numeric_cols, 'Category')], id.var='Category')

# change the width and height option
options(repr.plot.width=10, repr.plot.height=8) 
box_plot.C <- ggplot(data = hepc.mC, aes(x=variable, y=value)) + 
            geom_boxplot(aes_string(fill='Category')) + 
            facet_wrap( ~ variable, scales='free') + 
            xlab('Variable') + ylab('Values')
box_plot.C

# melt the dataframe into suitable shape for plotting
hepc.mgc <- melt(hepc[, c(numeric_cols, 'general_category')], id.var='general_category')

# change the width and height option
options(repr.plot.width=10, repr.plot.height=8) 
box_plot.gc <- ggplot(data = hepc.mgc, aes(x=variable, y=value)) + 
            geom_boxplot(aes_string(fill='general_category')) + 
            facet_wrap( ~ variable, scales='free') + 
            xlab('Variable') + ylab('Values')
box_plot.gc 

# melt the dataframe into suitable shape for plotting
hepc.mgc <- melt(hepc[, c(numeric_cols, 'Sex')], id.var='Sex')

# change the width and height option
options(repr.plot.width=10, repr.plot.height=8) 
box_plot.gc <- ggplot(data = hepc.mgc, aes(x=variable, y=value)) + 
            geom_boxplot(aes_string(fill='Sex')) + 
            facet_wrap( ~ variable, scales='free') + 
            xlab('Variable') + ylab('Values')
box_plot.gc 

hepc.summary_gc <- describeBy(hepc, "general_category", mat=TRUE)
hepc.summary_gc[, c('group1', 'median', 'min', 'max', 'skew')]

hepc.correlation <- round(cor(hepc[, numeric_cols]),2)

melted_cormat <- melt(hepc.correlation)

# Reorder the correlation matrix
cormat <- reorder_cormat(hepc.correlation)
upper_tri <- get_upper_tri(hepc.correlation)

# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- (ggplot(melted_cormat, aes(Var2, Var1, fill = value)) + 
              geom_tile(color = 'white') +
              scale_fill_gradient2(low = 'blue', high = 'red', mid = 'white', 
                                   midpoint = 0, limit = c(-1,1), space = 'Lab', 
                                   name='Pearson\nCorrelation') +
              theme_minimal() +
              theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                               size = 12, hjust = 1)) +
              coord_fixed()
             )

ggheatmap + 
geom_text(aes(Var2, Var1, label = value), color = 'black', size = 4) +
theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.grid.major = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.ticks = element_blank(),
  legend.justification = c(1, 0),
  legend.position = c(0.6, 0.7),
  legend.direction = 'horizontal')+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                title.position = 'top', title.hjust = 0.5))

sort(rowSums(abs(cormat)))

cols.1 <- c('ALP', 'AST', 'CREA', 'ALB')
cols.2 <- c('CHE', 'ALB', 'AST', 'GGT', 'CHOL')
cols.3 <- c('CREA', 'ALT', 'ALP', 'PROT', 'BIL')
cols.4 <- c('ALP', 'ALT', 'AST', 'GGT', 'CREA')

pairs(hepc.sc[, cols.1], pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))], oma=c(3, 3, 3, 15))
par(xpd=TRUE)
legend("bottomright", fill = c('blue', 'red'), legend = c(levels(hepc$general_category)), cex=0.7)

pairs(hepc.sc[, cols.2], pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))], oma=c(3, 3, 3, 15))
par(xpd=TRUE)
legend("bottomright", fill = c('blue', 'red'), legend = c(levels(hepc$general_category)), cex=0.7)

pairs(hepc.sc[, cols.3], pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))], oma=c(3, 3, 3, 15))
par(xpd=TRUE)
legend("bottomright", fill = c('blue', 'red'), legend = c(levels(hepc$general_category)), cex=0.7)

pairs(hepc.sc[, cols.4], pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))], oma=c(3, 3, 3, 15))
par(xpd=TRUE)
legend("bottomright", fill = c('blue', 'red'), legend = c(levels(hepc$general_category)), cex=0.7)

selected_cols

hepc.pca.selected_cols <- princomp(hepc.sc[, selected_cols])
hepc.pca.selected_cols
hepc.pca.selected_cols$loadings

# plot all the pc
pairs(hepc.pca.selected_cols$scores, pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))])

# plot pc1 against pc3
class_color <- c("red","blue")
plot(hepc.pca.selected_cols$scores[,1], hepc.pca.selected_cols$scores[,3], 
col=class_color[unclass(hepc$general_category)], 
cex=1.2, lwd=2, pch=0, 
xlab="PC1(10%)", ylab="PC3(10%)")

legend(-2, -7, pch=rep(0,4), pt.cex=1.2, pt.lwd=2,
col=class_color, c("Blood Donor", "Hepatisis" ))

abline(h=0,v=0, lty=3)

biplot(hepc.pca.selected_cols)

cols.1

hepc.pca.cols.1 <- princomp(hepc.sc[, cols.1])
hepc.pca.cols.1
hepc.pca.cols.1$loadings

# plot all the pc
pairs(hepc.pca.cols.1$scores, pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))])

# plot pc1 against pc3
class_color <- c('blue','red')
plot(hepc.pca.cols.1$scores[,2], hepc.pca.cols.1$scores[,3], 
col=class_color[unclass(hepc$general_category)], 
cex=1.2, lwd=2, pch=0, 
xlab='PC2(25%)', ylab='PC3(25%)')

legend(-2, -7, pch=rep(0,4), pt.cex=1.2, pt.lwd=2,
col=class_color, c('Blood Donor', 'Hepatisis' ))

abline(h=0,v=0, lty=3)

biplot(hepc.pca.cols.1)

hepc.pca.cols.2 <- princomp(hepc.sc[, cols.2])
hepc.pca.cols.2
hepc.pca.cols.2$loadings

# plot all the pc
pairs(hepc.pca.cols.2$scores, pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))])

# plot pc1 against pc3
class_color <- c('blue','red')
plot(hepc.pca.cols.2$scores[,2], hepc.pca.cols.2$scores[,3], 
col=class_color[unclass(hepc$general_category)], 
cex=1.2, lwd=2, pch=0, 
xlab='PC2(20%)', ylab='PC3(20%)')

legend(4, 2, pch=rep(0,4), pt.cex=1.2, pt.lwd=2,
col=class_color, c('Blood Donor', 'Hepatisis' ))

abline(h=0,v=0, lty=3)

biplot(hepc.pca.cols.2)

hepc.pca.cols.3 <- princomp(hepc.sc[, cols.3])
hepc.pca.cols.3
hepc.pca.cols.3$loadings

# plot all the pc
pairs(hepc.pca.cols.3$scores, pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))])

# plot pc1 against pc3
class_color <- c('blue','red')
plot(hepc.pca.cols.3$scores[,1], hepc.pca.cols.3$scores[,2], 
col=class_color[unclass(hepc$general_category)], 
cex=1.2, lwd=2, pch=0, 
xlab='PC1(20%)', ylab='PC2(20%)')

legend(-5, 6, pch=rep(0,4), pt.cex=1.2, pt.lwd=2,
col=class_color, c('Blood Donor', 'Hepatisis' ))

abline(h=0,v=0, lty=3)

biplot(hepc.pca.cols.3)

hepc.pca.cols.4 <- princomp(hepc.sc[, cols.4])
hepc.pca.cols.4
hepc.pca.cols.4$loadings

# plot all the pc
pairs(hepc.pca.cols.4$scores, pch=0, col=c('blue', 'red')[unclass(as.factor(hepc$general_category))])

# plot pc1 against pc3
class_color <- c('blue','red')
plot(hepc.pca.cols.4$scores[,1], hepc.pca.cols.4$scores[,4], 
col=class_color[unclass(hepc$general_category)], 
cex=1.2, lwd=2, pch=0, 
xlab='PC1(20%)', ylab='PC4(20%)')

legend(7, 0, pch=rep(0,4), pt.cex=1.2, pt.lwd=2,
col=class_color, c('Blood Donor', 'Hepatisis' ))

abline(h=0,v=0, lty=3)

biplot(hepc.pca.cols.4)

set.seed(1)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(hepc[, 2:13], hepc$general_category, size=c(1:12),  rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))

chisq.test(hepc$general_category, hepc$Sex)

table(hepc[, c('general_category', 'Sex')])

f_test_result <- lapply(hepc[, c(-1, -3, -14)], function(x) var.test(x~hepc$general_category)$p.value)
f_test_result <- data.frame(numeric_cols, matrix(f_test_result))
names(f_test_result) <- c('column', 'p_value')
f_test_result$p_value <- as.numeric(f_test_result$p_value)
f_test_result[order(f_test_result$p_value),]

colors <- c('blue', 'red')[as.numeric(hepc$general_category)]
scatterplot3d(x=hepc$AST, y=hepc$ALP, z=hepc$ALT,  color=colors)


fig <- plot_ly(hepc, x= ~AST, y= ~ALP, z= ~ALT, color=~general_category, colors= c('blue', 'red'))
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene= list(xaxis = list(title='AST'),
                                  yaxis = list(title='ALP'),
                                  zaxis = list(title='ALT')))
fig

htmlwidgets::saveWidget(as_widget(fig), "index.html")



library(tm)
library(proxy)
library(RTextTools)
library(fpc)   
library(wordcloud)
library(cluster)
library(tm)
library(stringi)
library(FactoMineR)

setwd("~/Documents/RStudioProjects/Movie-Script-Unsupervised-Learning-Methods-Analyses/choosenMoviesTest")
dir = DirSource(paste(getwd(), sep=""), encoding = "UTF-8")
#corpus = Corpus(dir, readerControl=list(reader=readPDF))
corpus = Corpus(dir)
summary(corpus)


ndocs <- length(corpus)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.01
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * .5
dtm = DocumentTermMatrix(corpus,
                         control = list(
                           stopwords = TRUE, 
                           wordLengths=c(4, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           #stemming = T,
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))
#dtm <- dtm[, names(head(sort(colSums(as.matrix(dtm))), 400))]
#dtm <- dtm[, names(sort(colSums(as.matrix(dtm))))]
#print(as.matrix(dtm))
setwd("~/Documents/RStudioProjects/Movie-Script-Unsupervised-Learning-Methods-Analyses/choosenMoviesOutput")
write.csv((as.matrix(dtm)), "test1.csv")
#head(sort(as.matrix(dtm)[18,], decreasing = TRUE), n=15)
dtm.matrix = as.matrix(dtm)
#wordcloud(colnames(dtm.matrix), dtm.matrix[28, ], max.words = 20)

inspect(dtm)

dtm <- weightTfIdf(dtm, normalize = TRUE)
dtm.matrix = as.matrix(dtm)
#wordcloud(colnames(dtm.matrix), dtm.matrix[28, ], max.words = 20)
#inspect(dtm)
write.csv((as.matrix(dtm)), "test2.csv")

head(sort(as.matrix(dtm)[1,], decreasing = TRUE), n=15)
#wordcloud(colnames(dtm.matrix), dtm.matrix[3, ], max.words = 200)



setwd("~/Documents/RStudioProjects/Movie-Script-Unsupervised-Learning-Methods-Analyses/choosenMoviesTest")
m  <- as.matrix(dtm)
# # # m <- m[1:2, 1:3]
distMatrix <- dist(m, method="euclidean")
#print(distMatrix)
#distMatrix <- dist(m, method="cosine")
#print(distMatrix)


groups <- hclust(distMatrix,method="ward.D")
plot(groups, cex=0.9, hang=-1)
rect.hclust(groups, k=5)
?FactoMineR






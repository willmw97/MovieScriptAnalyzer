# Movie Script Unsupervised Learning Methods Analyses

- details in the power point presentation (Movie Scripts.pptx) and word document report (Analysis of Movie Scripts.docx)


# Motivation
We wanted to see what are the most spoken words and prevailing emotions in movies. Because of this, we decided to do text mining on Movie Scripts and analyze text features. 

# Summary

We found that one of the largest emotions that are found in movies is trust. We also found that common genre movies do tend to cluster around each other (for example kids’ movies). We expected that similar movie genres would cluster together; we did not, however, get such results.

# Methods:
- Clustering (hclust and kmeans)
- Association Rules
- Sentiment Analysis 
- Frequency Plots


Upload Movies
-------------

``` r
path_85 = "~/OneDrive - MNSCU/myGithub/Unsupervised_Learning/Movie-Script-Unsupervised-Learning-Methods-Analyses/85_movies"
dir_85 = DirSource(paste(path_85, sep=""), encoding = "UTF-8")
corpus_85 = Corpus(dir_85)
head(summary(corpus_85))
```

    ##                          Length Class             Mode
    ## 500-Days-of-Summer.txt   2      PlainTextDocument list
    ## A-Few-Good-Men.txt       2      PlainTextDocument list
    ## A-Prayer-Before-Dawn.txt 2      PlainTextDocument list
    ## Abyss,-The.txt           2      PlainTextDocument list
    ## Alien.txt                2      PlainTextDocument list
    ## American-Outlaws.txt     2      PlainTextDocument list

``` r
tail(summary(corpus_85))
```

    ##                                   Length Class             Mode
    ## Tomorrow-Never-Dies.txt           2      PlainTextDocument list
    ## Vanilla-Sky.txt                   2      PlainTextDocument list
    ## Visitor,-The.txt                  2      PlainTextDocument list
    ## War-of-the-Worlds.txt             2      PlainTextDocument list
    ## When-a-Stranger-Calls.txt         2      PlainTextDocument list
    ## X-Files-Fight-the-Future,-The.txt 2      PlainTextDocument list

``` r
ndocs <- length(corpus_85)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.15
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * .5
dtm_85 = DocumentTermMatrix(corpus_85,
                         control = list(
                           stopwords = T, 
                           wordLengths=c(4, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           #stemming = T,
                           #removeWords("bateman"),
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))
```

``` r
#dtm <- dtm[, names(head(sort(colSums(as.matrix(dtm))), 400))]
#dtm <- dtm[, names(sort(colSums(as.matrix(dtm))))]
dtm.matrix_85 = as.matrix(dtm_85)
dim(dtm.matrix_85)
```

    ## [1]   85 3182

Clustering
----------

``` r
##### Preparing for clustering ######
```

``` r
#Clustering words
tdm_85 = t(dtm_85)
#Removing sparse terms
tdm_no_sparse_85 = removeSparseTerms(tdm_85, sparse = .99)
#Cluster
tdm.mat_85  <- as.matrix(tdm_no_sparse_85)
#First 20 most frequent words
word.freq = sort(rowSums(tdm.mat_85), decreasing = T)
barplot(word.freq[1:20], cex.names = .8)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
#Next 20 most frequent words
barplot(word.freq[21:40], cex.names = .8)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-6-2.png)

``` r
#Next 20 most frequent words
barplot(word.freq[41:60], cex.names = .8)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-6-3.png)

The most common words appear to be: <b>fuckin, ship, captain, train,
elevator, dances, bridge, agent, hotel, guards,</b> <br> summer, horse,
river, truck, court, fires, king, kicking, sister, york, <br> progress,
dawn, prison, trees, cabin, lieutenant, bird <br>

We are not considering names or words that relate to directions for
actors in the movie script; there are still some left out even after
cleaning all the 85 scripts in python to generate “spoken words”

### Clustering Words

``` r
#Clustering words: words as terms and movies as documents: T-D
clust.tdm.matrix_85 = as.matrix(tdm_85)
clust.tdm.matrix_85[clust.tdm.matrix_85>1] = 1
```

``` r
#Using Agnes
tdm.agnes.ward_85 = agnes(clust.tdm.matrix_85, metric = "Jaccard", method = "ward")
```

``` r
#Plotting Agnes cluster
plot(tdm.agnes.ward_85,cex = 0.7, which.plots = 2)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
#Using hclust
tdm.movie.dist_85 = dist(clust.tdm.matrix_85, method = "Jaccard")
```

``` r
#Plotting hlust
tdm.movie.clust_85 = hclust(tdm.movie.dist_85, method = "ward.D")
plot(tdm.movie.clust_85, cex = .7)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-11-1.png)

``` r
words.groups_85 = cutree(tdm.movie.clust_85, k=25)
table(words.groups_85)
```

    ## words.groups_85
    ##   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18 
    ## 106 106 190 134 142 244 101 117 148  81 292  62  62 138  60 121 147  84 
    ##  19  20  21  22  23  24  25 
    ## 150 142 107  52 123 179  94

#### K-means

``` r
#Optimal number of clusters
fviz_nbclust(clust.tdm.matrix_85,kmeans,k.max=15,method="silhouette")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-13-1.png)

``` r
fviz_nbclust(tdm.mat_85,kmeans,k.max=15,method="wss")  
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
#fviz_nbclust(clust.tdm.matrix_85,kmeans,k.max=15,method = "gap_stat")
```

``` r
shooby_85_words = kmeans(clust.tdm.matrix_85,3)
table(shooby_85_words$cluster)
```

    ## 
    ##    1    2    3 
    ##  856 1061 1265

``` r
fviz_cluster(shooby_85_words,data = clust.tdm.matrix_85)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-17-1.png)

### Clustering Movies

``` r
#Clustering movies: movies as documents and words as terms: D-T
clust.dtm.matrix_85 = dtm.matrix_85
clust.dtm.matrix_85[clust.dtm.matrix_85>1] = 1

#Clustering words: words as terms and movies as documents: T-D
clust.tdm.matrix_85 = as.matrix(tdm_85)
clust.tdm.matrix_85[clust.tdm.matrix_85>1] = 1
```

``` r
#Using Agnes
dtm.agnes.ward_85 = agnes(clust.dtm.matrix_85, metric = "Jaccard", method = "ward")
```

``` r
#Plotting Agnes cluster
plot(dtm.agnes.ward_85,cex = 0.7, which.plots = 2)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-20-1.png)

``` r
#Using hclust
dtm.movie.dist_85 = dist(clust.dtm.matrix_85, method = "Jaccard")
```

``` r
#Plotting hlust
dtm.movie.clust_85 = hclust(dtm.movie.dist_85, method = "ward.D")
plot(dtm.movie.clust_85, cex = .7)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-22-1.png)

``` r
#Plotting hclust with movie genre labels
read.csv("emotions_final.csv") -> label
plot(dtm.movie.clust_85, cex = .7, labels = label$Genre)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-23-1.png)

``` r
movie.groups_85 = cutree(dtm.movie.clust_85, k=3)
table(movie.groups_85, label$Genre)
```

    ##                
    ## movie.groups_85 Action Adventure Comedy Crime Family Fantasy Horror
    ##               1      4         0      7     3      3       0      1
    ##               2      6         1      3     9      1       0      6
    ##               3      2         0      1     0      0       2      1
    ##                
    ## movie.groups_85 Romance Sci-Fi Thriller War
    ##               1       5      5        4   0
    ##               2       4      2        9   1
    ##               3       1      2        2   0

#### K-means

``` r
#Optimal number of clusters
fviz_nbclust(clust.dtm.matrix_85,kmeans,k.max=15,method="silhouette")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-25-1.png)

``` r
fviz_nbclust(clust.dtm.matrix_85,kmeans,k.max=15,method="wss")  
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-26-1.png)

``` r
fviz_nbclust(clust.dtm.matrix_85,kmeans,k.max=15,method = "gap_stat")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-27-1.png)

``` r
shooby_85 = kmeans(clust.dtm.matrix_85,3)
table(shooby_85$cluster, label$Genre)
```

    ##    
    ##     Action Adventure Comedy Crime Family Fantasy Horror Romance Sci-Fi
    ##   1      6         0      2     8      1       0      3       3      4
    ##   2      4         0      7     2      3       2      2       6      4
    ##   3      2         1      2     2      0       0      3       1      1
    ##    
    ##     Thriller War
    ##   1        2   0
    ##   2        5   0
    ##   3        8   1

``` r
fviz_cluster(shooby_85,data = clust.dtm.matrix_85)
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-29-1.png)

### Labels

### Association Rules: Movies as transactions and words as products purchased

``` r
#Our clust.dtm.matrix
movie.trans_85 = as(clust.dtm.matrix_85, "transactions")
#With support .3 and confidence .8 -> 39 rules found
movie.rules_85 = apriori(movie.trans_85, parameter = list(supp = .2, conf =.9))
```

    ## Apriori
    ## 
    ## Parameter specification:
    ##  confidence minval smax arem  aval originalSupport maxtime support minlen
    ##         0.9    0.1    1 none FALSE            TRUE       5     0.2      1
    ##  maxlen target   ext
    ##      10  rules FALSE
    ## 
    ## Algorithmic control:
    ##  filter tree heap memopt load sort verbose
    ##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
    ## 
    ## Absolute minimum support count: 17 
    ## 
    ## set item appearances ...[0 item(s)] done [0.00s].
    ## set transactions ...[3182 item(s), 85 transaction(s)] done [0.01s].
    ## sorting and recoding items ... [2271 item(s)] done [0.00s].
    ## creating transaction tree ... done [0.00s].
    ## checking subsets of size 1 2 3 4 5 6 7 done [0.96s].
    ## writing ... [37948 rule(s)] done [0.09s].
    ## creating S4 object  ... done [0.05s].

About 38,000 rules generates

``` r
plot(movie.rules_85, "grouped")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-33-1.png)

``` r
plot(movie.rules_85, "scatter", jitter = 0, interactive = F)
```

    ## Warning in plot.rules(movie.rules_85, "scatter", jitter = 0, interactive
    ## = F): The parameter interactive is deprecated. Use engine='interactive'
    ## instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-34-1.png) As
seen above in the scatter plot, it’d be best to take rules that are the
redest (highest lift value) and are in the top right corner (highest
confidence and support). Let’s expore some through ruleExplorer() to
determine about 10 best quality rules

Now, we are going to create subsets on rules with RHS being the most
frequent word used

3 rules for 10 most frequent words: <br> <b>fuckin, ship, captain,
train, elevator, dances, bridge, agent, hotel, guards,</b> <br>

Make plots interactive and take screenshots
-------------------------------------------

``` r
movie.rule_1 = sort(movie.rule_1, by = "lift")
inspect(movie.rule_1)
```

    ##      lhs                      rhs      support   confidence lift     count
    ## [1]  {waitress,fifteen}    => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [2]  {friday,dollar}       => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [3]  {friday,parking}      => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [4]  {bullshit,nothin}     => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [5]  {treat,television}    => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [6]  {shower,suck}         => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [7]  {fucked,sentence}     => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [8]  {beer,society}        => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [9]  {tied,neighborhood}   => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [10] {storms,fucked}       => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [11] {nuts,fucked}         => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [12] {fucked,accident}     => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [13] {fucked,clutching}    => {fuckin} 0.1176471 0.8333333  2.951389 10   
    ## [14] {bullshit,tied}       => {fuckin} 0.1529412 0.8125000  2.877604 13   
    ## [15] {talked,neighborhood} => {fuckin} 0.1411765 0.8000000  2.833333 12   
    ## [16] {fucked,guilty}       => {fuckin} 0.1411765 0.8000000  2.833333 12

``` r
write(movie.rule_1,
      file = "association_rules.csv",
      sep = ",",
      quote = TRUE,
      row.names = FALSE)
```

``` r
plot(movie.rule_1, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_1, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-41-1.png)

``` r
movie.rule_2 = sort(movie.rule_2, by = "lift")
inspect(movie.rule_2)
```

    ##      lhs                    rhs    support   confidence lift     count
    ## [1]  {directly,ships}    => {ship} 0.1411765 0.9230769  3.566434 12   
    ## [2]  {ships,effect}      => {ship} 0.1411765 0.9230769  3.566434 12   
    ## [3]  {cutting,ships}     => {ship} 0.1411765 0.9230769  3.566434 12   
    ## [4]  {ships}             => {ship} 0.1647059 0.8750000  3.380682 14   
    ## [5]  {ships,surface}     => {ship} 0.1529412 0.8666667  3.348485 13   
    ## [6]  {safety,ships}      => {ship} 0.1411765 0.8571429  3.311688 12   
    ## [7]  {blast,ships}       => {ship} 0.1411765 0.8571429  3.311688 12   
    ## [8]  {ships,landing}     => {ship} 0.1411765 0.8571429  3.311688 12   
    ## [9]  {floating,surface}  => {ship} 0.1411765 0.8000000  3.090909 12   
    ## [10] {directly,floating} => {ship} 0.1411765 0.8000000  3.090909 12

``` r
plot(movie.rule_2, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_2, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-43-1.png)

``` r
movie.rule_3 = sort(movie.rule_3, by = "lift")
inspect(movie.rule_3)
```

    ##      lhs                     rhs       support   confidence lift     count
    ## [1]  {climbing,recovered} => {captain} 0.1176471 1.0000000  3.400000 10   
    ## [2]  {process,reports}    => {captain} 0.1294118 1.0000000  3.400000 11   
    ## [3]  {lever,fortune}      => {captain} 0.1294118 0.9166667  3.116667 11   
    ## [4]  {reports,south}      => {captain} 0.1294118 0.9166667  3.116667 11   
    ## [5]  {panel,fortune}      => {captain} 0.1294118 0.9166667  3.116667 11   
    ## [6]  {route,sake}         => {captain} 0.1294118 0.9166667  3.116667 11   
    ## [7]  {crashing,recovered} => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [8]  {explode,recovered}  => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [9]  {dropping,recovered} => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [10] {avoid,engines}      => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [11] {armed,ocean}        => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [12] {dives,ocean}        => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [13] {safety,strains}     => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [14] {reports,route}      => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [15] {reports,panel}      => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [16] {leap,fortune}       => {captain} 0.1176471 0.9090909  3.090909 10   
    ## [17] {pulse,panel}        => {captain} 0.1176471 0.9090909  3.090909 10

``` r
plot(movie.rule_3, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_3, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-45-1.png)

``` r
movie.rule_4 = sort(movie.rule_4, by = "lift")
inspect(movie.rule_4)
```

    ##      lhs                          rhs     support   confidence lift    
    ## [1]  {atmosphere,public}       => {train} 0.1764706 1.0000000  2.125000
    ## [2]  {public,rows}             => {train} 0.1647059 1.0000000  2.125000
    ## [3]  {intense,yelling}         => {train} 0.1647059 1.0000000  2.125000
    ## [4]  {bench,drag}              => {train} 0.1882353 0.9411765  2.000000
    ## [5]  {hardly,noticed}          => {train} 0.1764706 0.9375000  1.992188
    ## [6]  {intense,thousands}       => {train} 0.1647059 0.9333333  1.983333
    ## [7]  {files,visible}           => {train} 0.1647059 0.9333333  1.983333
    ## [8]  {familiar,slipping}       => {train} 0.1647059 0.9333333  1.983333
    ## [9]  {passengers,switch}       => {train} 0.1529412 0.9285714  1.973214
    ## [10] {liar,possibly}           => {train} 0.1529412 0.9285714  1.973214
    ## [11] {confusion,pavement}      => {train} 0.1529412 0.9285714  1.973214
    ## [12] {visible,offices}         => {train} 0.1529412 0.9285714  1.973214
    ## [13] {willing,ghost}           => {train} 0.1529412 0.9285714  1.973214
    ## [14] {sidewalk,workers}        => {train} 0.1529412 0.9285714  1.973214
    ## [15] {knowing,workers}         => {train} 0.1529412 0.9285714  1.973214
    ## [16] {lovely,particular}       => {train} 0.1529412 0.9285714  1.973214
    ## [17] {accept,national}         => {train} 0.1529412 0.9285714  1.973214
    ## [18] {national,visible}        => {train} 0.1529412 0.9285714  1.973214
    ## [19] {rows,smashing}           => {train} 0.1529412 0.9285714  1.973214
    ## [20] {common,alley}            => {train} 0.1529412 0.9285714  1.973214
    ## [21] {common,split}            => {train} 0.1529412 0.9285714  1.973214
    ## [22] {directions,professional} => {train} 0.1529412 0.9285714  1.973214
    ## [23] {envelope,alley}          => {train} 0.1529412 0.9285714  1.973214
    ## [24] {intense,envelope}        => {train} 0.1529412 0.9285714  1.973214
    ## [25] {sheet,envelope}          => {train} 0.1529412 0.9285714  1.973214
    ## [26] {knocked,envelope}        => {train} 0.1529412 0.9285714  1.973214
    ## [27] {paces,envelope}          => {train} 0.1529412 0.9285714  1.973214
    ## [28] {conference,yesterday}    => {train} 0.1529412 0.9285714  1.973214
    ## [29] {possibly,respect}        => {train} 0.1529412 0.9285714  1.973214
    ## [30] {knocked,relieved}        => {train} 0.1529412 0.9285714  1.973214
    ## [31] {tension,mass}            => {train} 0.1529412 0.9285714  1.973214
    ##      count
    ## [1]  15   
    ## [2]  14   
    ## [3]  14   
    ## [4]  16   
    ## [5]  15   
    ## [6]  14   
    ## [7]  14   
    ## [8]  14   
    ## [9]  13   
    ## [10] 13   
    ## [11] 13   
    ## [12] 13   
    ## [13] 13   
    ## [14] 13   
    ## [15] 13   
    ## [16] 13   
    ## [17] 13   
    ## [18] 13   
    ## [19] 13   
    ## [20] 13   
    ## [21] 13   
    ## [22] 13   
    ## [23] 13   
    ## [24] 13   
    ## [25] 13   
    ## [26] 13   
    ## [27] 13   
    ## [28] 13   
    ## [29] 13   
    ## [30] 13   
    ## [31] 13

``` r
plot(movie.rule_4, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_4, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-47-1.png)

``` r
movie.rule_5 = sort(movie.rule_5, by = "lift")
inspect(movie.rule_5)
```

    ##     lhs                     rhs        support   confidence lift     count
    ## [1] {conference,earlier} => {elevator} 0.1882353 0.9411765  2.222222 16   
    ## [2] {assistant,pockets}  => {elevator} 0.1882353 0.9411765  2.222222 16   
    ## [3] {necessary,taxi}     => {elevator} 0.1882353 0.8888889  2.098765 16   
    ## [4] {allowed,tray}       => {elevator} 0.1882353 0.8888889  2.098765 16   
    ## [5] {earlier,truck}      => {elevator} 0.2117647 0.8571429  2.023810 18   
    ## [6] {catching,glimpse}   => {elevator} 0.2000000 0.8500000  2.006944 17   
    ## [7] {clearly,sidewalk}   => {elevator} 0.2000000 0.8500000  2.006944 17   
    ## [8] {offers,tray}        => {elevator} 0.2000000 0.8500000  2.006944 17

``` r
plot(movie.rule_5, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_5, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-49-1.png)

``` r
movie.rule_6 = sort(movie.rule_6, by = "lift")
inspect(movie.rule_6)
```

    ##     lhs                  rhs      support    confidence lift     count
    ## [1] {awkwardly,sides} => {dances} 0.10588235 0.8181818  4.090909 9    
    ## [2] {saturday,wallet} => {dances} 0.09411765 0.8000000  4.000000 8    
    ## [3] {fate,knocking}   => {dances} 0.09411765 0.8000000  4.000000 8    
    ## [4] {peace,grips}     => {dances} 0.09411765 0.8000000  4.000000 8

``` r
plot(movie.rule_6, "graph", interactive = T)
```

    ## Warning in plot.rules(movie.rule_6, "graph", interactive = T): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

``` r
movie.rule_7 = sort(movie.rule_7, by = "lift")
inspect(movie.rule_7)
```

    ##      lhs                                  rhs      support   confidence
    ## [1]  {races,iron,surface}              => {bridge} 0.1764706 1.0000000 
    ## [2]  {base,lowers,surface}             => {bridge} 0.1882353 0.9411765 
    ## [3]  {doubt,ends,uses}                 => {bridge} 0.1882353 0.9411765 
    ## [4]  {sparks,surface}                  => {bridge} 0.1764706 0.9375000 
    ## [5]  {iron,surface}                    => {bridge} 0.1764706 0.9375000 
    ## [6]  {surface,climbing,releases}       => {bridge} 0.1764706 0.9375000 
    ## [7]  {inches,flashes,frantically}      => {bridge} 0.1764706 0.9375000 
    ## [8]  {distant,chin,lowers}             => {bridge} 0.1764706 0.9375000 
    ## [9]  {base,lowers,peers}               => {bridge} 0.1764706 0.9375000 
    ## [10] {directly,inches,surface,flashes} => {bridge} 0.1764706 0.9375000 
    ## [11] {races,base,lowers,surface}       => {bridge} 0.1764706 0.9375000 
    ## [12] {base,flat,lowers,race}           => {bridge} 0.1764706 0.9375000 
    ##      lift     count
    ## [1]  2.428571 15   
    ## [2]  2.285714 16   
    ## [3]  2.285714 16   
    ## [4]  2.276786 15   
    ## [5]  2.276786 15   
    ## [6]  2.276786 15   
    ## [7]  2.276786 15   
    ## [8]  2.276786 15   
    ## [9]  2.276786 15   
    ## [10] 2.276786 15   
    ## [11] 2.276786 15   
    ## [12] 2.276786 15

``` r
plot(movie.rule_7, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_7, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-53-1.png)

``` r
movie.rule_8 = sort(movie.rule_8, by = "lift")
inspect(movie.rule_8)
```

    ##      lhs                            rhs     support   confidence lift    
    ## [1]  {agents,crime}              => {agent} 0.1529412 1.0000000  3.148148
    ## [2]  {knowing,agents}            => {agent} 0.1529412 0.9285714  2.923280
    ## [3]  {agents}                    => {agent} 0.1882353 0.8888889  2.798354
    ## [4]  {agents,dangerous}          => {agent} 0.1529412 0.8666667  2.728395
    ## [5]  {familiar,agents}           => {agent} 0.1529412 0.8666667  2.728395
    ## [6]  {elevator,bullet,struggle}  => {agent} 0.1529412 0.8666667  2.728395
    ## [7]  {elevator,gear,bullet}      => {agent} 0.1529412 0.8666667  2.728395
    ## [8]  {growing,bullet,blocks}     => {agent} 0.1529412 0.8666667  2.728395
    ## [9]  {elevator,gear,struggle}    => {agent} 0.1647059 0.8235294  2.592593
    ## [10] {gear,surveillance}         => {agent} 0.1529412 0.8125000  2.557870
    ## [11] {dozen,government}          => {agent} 0.1529412 0.8125000  2.557870
    ## [12] {bringing,dozen,government} => {agent} 0.1529412 0.8125000  2.557870
    ## [13] {gate,passenger,crime}      => {agent} 0.1529412 0.8125000  2.557870
    ## [14] {gear,bullet,inch}          => {agent} 0.1529412 0.8125000  2.557870
    ## [15] {freezes,bullet,inch}       => {agent} 0.1529412 0.8125000  2.557870
    ## [16] {bullet,scans,inch}         => {agent} 0.1529412 0.8125000  2.557870
    ## [17] {gate,bullet,inch}          => {agent} 0.1529412 0.8125000  2.557870
    ## [18] {elevator,growing,struggle} => {agent} 0.1529412 0.8125000  2.557870
    ## [19] {gate,lined,passenger}      => {agent} 0.1529412 0.8125000  2.557870
    ## [20] {address,dozen,fires}       => {agent} 0.1529412 0.8125000  2.557870
    ## [21] {address,dozen,whispers}    => {agent} 0.1529412 0.8125000  2.557870
    ## [22] {growing,bullet,scans}      => {agent} 0.1529412 0.8125000  2.557870
    ## [23] {growing,bullet,dozen}      => {agent} 0.1529412 0.8125000  2.557870
    ## [24] {bullet,dozen,forces}       => {agent} 0.1529412 0.8125000  2.557870
    ## [25] {copy,gear,monitor}         => {agent} 0.1529412 0.8125000  2.557870
    ## [26] {gear,charges,monitor}      => {agent} 0.1529412 0.8125000  2.557870
    ## [27] {copy,dozen,fires}          => {agent} 0.1529412 0.8125000  2.557870
    ## [28] {dozen,fires,struggle}      => {agent} 0.1529412 0.8125000  2.557870
    ## [29] {bringing,dozen,forces}     => {agent} 0.1529412 0.8125000  2.557870
    ## [30] {dozen,forces,match}        => {agent} 0.1529412 0.8125000  2.557870
    ##      count
    ## [1]  13   
    ## [2]  13   
    ## [3]  16   
    ## [4]  13   
    ## [5]  13   
    ## [6]  13   
    ## [7]  13   
    ## [8]  13   
    ## [9]  14   
    ## [10] 13   
    ## [11] 13   
    ## [12] 13   
    ## [13] 13   
    ## [14] 13   
    ## [15] 13   
    ## [16] 13   
    ## [17] 13   
    ## [18] 13   
    ## [19] 13   
    ## [20] 13   
    ## [21] 13   
    ## [22] 13   
    ## [23] 13   
    ## [24] 13   
    ## [25] 13   
    ## [26] 13   
    ## [27] 13   
    ## [28] 13   
    ## [29] 13   
    ## [30] 13

``` r
plot(movie.rule_8, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_8, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-55-1.png)

``` r
movie.rule_9 = sort(movie.rule_9, by = "lift")
inspect(movie.rule_9)
```

    ##      lhs                     rhs     support   confidence lift     count
    ## [1]  {address,shrugs}     => {hotel} 0.2235294 0.9047619  2.023810 19   
    ## [2]  {cash,shrugs}        => {hotel} 0.2117647 0.9000000  2.013158 18   
    ## [3]  {newspaper,shrugs}   => {hotel} 0.2117647 0.9000000  2.013158 18   
    ## [4]  {shrugs,excitement}  => {hotel} 0.2470588 0.8750000  1.957237 21   
    ## [5]  {pause,shrugs}       => {hotel} 0.2470588 0.8750000  1.957237 21   
    ## [6]  {rooms,shrugs}       => {hotel} 0.2352941 0.8695652  1.945080 20   
    ## [7]  {wine,shrugs}        => {hotel} 0.2235294 0.8636364  1.931818 19   
    ## [8]  {boyfriend,expect}   => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [9]  {bill,dollar}        => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [10] {appreciate,dollar}  => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [11] {dollar,expect}      => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [12] {appreciate,john}    => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [13] {appreciate,shocked} => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [14] {class,shrugs}       => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [15] {parking,shrugs}     => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [16] {brief,shrugs}       => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [17] {unable,shrugs}      => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [18] {rooms,shocked}      => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [19] {address,bunch}      => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [20] {shrugs,john}        => {hotel} 0.2117647 0.8571429  1.917293 18   
    ## [21] {earlier,shrugs}     => {hotel} 0.2117647 0.8571429  1.917293 18

``` r
plot(movie.rule_9, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_9, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-57-1.png)

``` r
movie.rule_10 = sort(movie.rule_10, by = "lift")
inspect(movie.rule_10)
```

    ##      lhs                           rhs      support   confidence lift    
    ## [1]  {gear,fires,drag}          => {guards} 0.2117647 0.9000000  2.185714
    ## [2]  {fires,drag,scans}         => {guards} 0.2117647 0.9000000  2.185714
    ## [3]  {dozen,rushing}            => {guards} 0.2235294 0.8636364  2.097403
    ## [4]  {ends,rushing}             => {guards} 0.2117647 0.8571429  2.081633
    ## [5]  {remain,charges}           => {guards} 0.2117647 0.8571429  2.081633
    ## [6]  {races,backwards}          => {guards} 0.2117647 0.8571429  2.081633
    ## [7]  {footsteps,drag}           => {guards} 0.2117647 0.8571429  2.081633
    ## [8]  {charges,drag}             => {guards} 0.2117647 0.8571429  2.081633
    ## [9]  {exhausted,roar,terrified} => {guards} 0.2117647 0.8571429  2.081633
    ## [10] {bullet,fires,drag}        => {guards} 0.2117647 0.8571429  2.081633
    ## [11] {punches,forces,drag}      => {guards} 0.2117647 0.8571429  2.081633
    ## [12] {gear,fires,knocking}      => {guards} 0.2117647 0.8571429  2.081633
    ##      count
    ## [1]  18   
    ## [2]  18   
    ## [3]  19   
    ## [4]  18   
    ## [5]  18   
    ## [6]  18   
    ## [7]  18   
    ## [8]  18   
    ## [9]  18   
    ## [10] 18   
    ## [11] 18   
    ## [12] 18

``` r
plot(movie.rule_10, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_10, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-59-1.png)

``` r
movie.rule_11 = sort(movie.rule_11, by = "lift")
inspect(movie.rule_11)
```

    ##      lhs                               rhs      support   confidence
    ## [1]  {gear,device,east}             => {tunnel} 0.1764706 1.0000000 
    ## [2]  {freezes,gear,device}          => {tunnel} 0.1764706 1.0000000 
    ## [3]  {gear,device,cracks}           => {tunnel} 0.1764706 1.0000000 
    ## [4]  {dozen,device,cracks}          => {tunnel} 0.1764706 1.0000000 
    ## [5]  {gear,device,explodes,cracks}  => {tunnel} 0.1764706 1.0000000 
    ## [6]  {dozen,device,explodes,cracks} => {tunnel} 0.1764706 1.0000000 
    ## [7]  {gear,dozen,device,cracks}     => {tunnel} 0.1764706 1.0000000 
    ## [8]  {gear,device,monitor}          => {tunnel} 0.1882353 0.9411765 
    ## [9]  {gear,climbing,device}         => {tunnel} 0.1882353 0.9411765 
    ## [10] {gear,signal,device}           => {tunnel} 0.1882353 0.9411765 
    ## [11] {device,east}                  => {tunnel} 0.1764706 0.9375000 
    ## [12] {slide,device}                 => {tunnel} 0.1764706 0.9375000 
    ## [13] {dangerous,device}             => {tunnel} 0.1764706 0.9375000 
    ## [14] {freezes,device}               => {tunnel} 0.1764706 0.9375000 
    ## [15] {gear,armed,device}            => {tunnel} 0.1764706 0.9375000 
    ## [16] {gear,anger,device}            => {tunnel} 0.1764706 0.9375000 
    ## [17] {gear,slide,device}            => {tunnel} 0.1764706 0.9375000 
    ## [18] {uses,climbing,device}         => {tunnel} 0.1764706 0.9375000 
    ## [19] {gear,dangerous,device}        => {tunnel} 0.1764706 0.9375000 
    ## [20] {gear,punches,device}          => {tunnel} 0.1764706 0.9375000 
    ## [21] {gear,switch,device}           => {tunnel} 0.1764706 0.9375000 
    ## [22] {gear,spinning,dropping}       => {tunnel} 0.1764706 0.9375000 
    ## [23] {gear,spinning,weight}         => {tunnel} 0.1764706 0.9375000 
    ## [24] {shifts,dozen,cracks}          => {tunnel} 0.1764706 0.9375000 
    ## [25] {gear,uses,climbing,device}    => {tunnel} 0.1764706 0.9375000 
    ## [26] {gear,shifts,dozen,cracks}     => {tunnel} 0.1764706 0.9375000 
    ## [27] {gear,device}                  => {tunnel} 0.2117647 0.9000000 
    ##      lift     count
    ## [1]  3.541667 15   
    ## [2]  3.541667 15   
    ## [3]  3.541667 15   
    ## [4]  3.541667 15   
    ## [5]  3.541667 15   
    ## [6]  3.541667 15   
    ## [7]  3.541667 15   
    ## [8]  3.333333 16   
    ## [9]  3.333333 16   
    ## [10] 3.333333 16   
    ## [11] 3.320312 15   
    ## [12] 3.320312 15   
    ## [13] 3.320312 15   
    ## [14] 3.320312 15   
    ## [15] 3.320312 15   
    ## [16] 3.320312 15   
    ## [17] 3.320312 15   
    ## [18] 3.320312 15   
    ## [19] 3.320312 15   
    ## [20] 3.320312 15   
    ## [21] 3.320312 15   
    ## [22] 3.320312 15   
    ## [23] 3.320312 15   
    ## [24] 3.320312 15   
    ## [25] 3.320312 15   
    ## [26] 3.320312 15   
    ## [27] 3.187500 18

``` r
plot(movie.rule_11, "graph", interactive = F)
```

    ## Warning in plot.rules(movie.rule_11, "graph", interactive = F): The
    ## parameter interactive is deprecated. Use engine='interactive' instead.

![](85_Movies_files/figure-markdown_github/unnamed-chunk-61-1.png)

\#Work on making rule subsets based on the most frequent word for
example. Relate to word.frequency bar graphs

\#Sentiment

Word cloud for all the movies
-----------------------------

``` r
mystopwords = c(stopwords("en"), "jack", "continued", "john")
```

``` r
check = Corpus(dir_85)
check = tm_map(check, removeWords, mystopwords)
```

``` r
ndocs <- length(check)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.15
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * .5
checkDTM = DocumentTermMatrix(check,
                         control = list(
                           wordLengths=c(4, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           #stemming = T,
                           #removeWords("bateman"),
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))
checkTDM = t(checkDTM)
check.mat = as.matrix(checkTDM)
```

``` r
word.freq_check = sort(rowSums(check.mat), decreasing = T)
barplot(word.freq_check[1:20], cex.names = .8, col = "light blue")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-66-1.png)

``` r
#Next 20 most frequent words
barplot(word.freq_check[21:40], cex.names = .8, col = "light blue")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-66-2.png)

``` r
#Next 20 most frequent words
barplot(word.freq_check[41:60], cex.names = .8, col = "light blue")
```

![](85_Movies_files/figure-markdown_github/unnamed-chunk-66-3.png)

``` r
dim(check.mat)
```

    ## [1] 3200   85
=======
>>>>>>> c4cbb64c091fd7437f604ec2ea2c0e3f59200b94

A scRNA-seq interface to bulk RNA-seq datasets

Ingredients:
- ARCHS4 database
- Poisson distance
- UNCURL for getting M, the cell archetypes

Methods:
1. Normalize all datasets so that they sum to 1.
2. Binarize the normalized expression level for each gene using the qualNorm binarization method - clustering into two clusters and separating by means.
3. Build a search index by Hamming distance.

How to deal with missing genes? Is a "skyline query" a good method?

Missing gene imputation? What kind of model to use?

Protein Atlas

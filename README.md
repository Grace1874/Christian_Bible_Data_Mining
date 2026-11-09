# Christian Bible Data Mining

## Project Overview
This project applies data mining techniques to the Bible Corpus dataset to discover textual patterns, classify Bible verses, and group verses based on word similarities.

## Dataset
- Dataset: Bible Corpus
- Text source: King James Version (KJV)
- File: `t_kjv.csv`
- Original source: Kaggle — Bible Corpus by oswinrh

Please obtain the dataset from its original Kaggle source and check its license before redistribution.

## Algorithms Implemented

### Association Rule Mining
1. Apriori
2. FP-Growth

### Decision Tree Classification
3. ID3-style Decision Tree
4. CART
5. C4.5-style Decision Tree

### Clustering
6. K-Means Clustering
7. Hierarchical Clustering

## Project Structure

```text
Christian_Bible_Data_Mining/
│
├── Code/
│   ├── 01_data_exploration.py
│   ├── 02_text_preprocessing.py
│   ├── 03_apriori.py
│   ├── 04_fp_growth.py
│   ├── 05_prepare_tree_data.py
│   ├── 06_id3.py
│   ├── 07_cart.py
│   ├── 08_kmeans.py
│   ├── 09_c45.py
│   └── 10_hierarchical.py
│
├── Dataset/
├── Results/
├── Report/
└── README.md
```

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- MLxtend

## Notes
- ID3 and C4.5 are simplified educational implementations.
- Hierarchical clustering uses a sample of 2,000 verses.
- Association rules describe textual word co-occurrences, not theological conclusions.

## Author
Grace1874

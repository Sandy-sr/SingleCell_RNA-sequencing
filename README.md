# Introduction
The project focused on utilizing unsupervised machine learning algorithms for the analysis of single-cell RNA sequencing (scRNA-seq) data. The goal was to cluster the data into 16 predefined clusters. RNA sequencing (RNA-seq) is a genetic technique that allows the detection and quantification of messenger RNA molecules in biological samples, providing valuable insights into cellular responses. In this clustering competition, the objective was to interpret the scRNA-seq dataset using various clustering algorithms and achieve the highest accuracy rate with one of the clustering models.

# Dataset
As part of the data preparation phase, the training dataset consists of 13,177 cell samples and 13,166 unique genes. In order to normalize the dataset before training the model, two pre-processing techniques were employed.

# Data Preparation
After loading the dataset, the initial step involved examining its shape and obtaining a descriptive summary using the describe() function. This allowed us to ensure that there were no missing values and gain additional insights necessary for subsequent data preprocessing steps. Two preprocessing libraries from Scikit-learn were utilized:

1) Correlation: Performed feature selection and removed highly correlated features using Correlation matrix. cols_to_drop = [1099, 1113, 1875, 2093, 2446, 2620, 2937, 3141, 4431, 9586, 10778], these columns where dropped before proceding to the MinMaxScalar.

2) MinMaxScaler: This library scales each feature within a specific range, transforming it accordingly. By independently scaling and translating each feature, it ensures that the values fall within the desired range set by the training data. The feature_range parameter was set to (0, 10) in the MinMaxScaler function.

3) PCA (Principal Component Analysis): PCA utilizes Singular Value Decomposition to reduce the dimensionality of the data and project it onto a lower-dimensional space. Prior to applying SVD, the input data is centered but not scaled for each feature. The n_components and random_state parameters were used in the PCA function, with n_components set to 100 and random_state set to 0.

Following the preprocessing steps mentioned above, incorporating the specified hyperparameters, the dataset was prepared. We then proceeded to train our model using various hyperparameter configurations to achieve the highest possible accuracy rate.

# Model Training
In our project, we experimented with various models and different combinations of preprocessing techniques. To evaluate the performance of the models, we used the silhouette score as our internal evaluation metric. This allowed us to assess the clustering quality before utilizing the API to obtain the final score, saving time during the process of trying out multiple combinations. After analyzing the results, we concluded that the Birch model performed the best among the models we tested. 

Birch, short for Balanced Iterative Reducing and Clustering using Hierarchies, is particularly suitable for handling large datasets. It first generates a compact summary that retains sufficient allocation information and then clusters the data summary instead of the entire dataset. Birch can be used in conjunction with other clustering algorithms, as it provides a summary that can be processed by them. Birch primarily supports metric characteristics, similar to how K-means handles features. A metric characteristic refers to values that can be represented in Euclidean space using explicit coordinates, excluding categorical variables. We conducted hyperparameter tuning, adjusting parameters such as branching_factor, n_clusters, and threshold, to obtain the best score for our model.

The optimized Birch model configuration we used was: Birch(n_clusters=16,threshold=3.5,branching_factor=400)

# Result
Our team Soul secured 3rd place in the competition with the following results,

* Birch Accuracy Result: % 0.8984336070338054
* Birch Silhouette Score: 0.116

# Conclusion
In our project, we explored various preprocessing combinations and employed different clustering techniques such as K-means, Agglomerative, and Birch. The preprocessing techniques included dimensionality reduction and minmax scaling. We utilized the silhouette score as our internal evaluation metric to assess the quality of clustering. After experimenting with different combinations, we achieved an impressive score of 89.84%, resulting in a significant rise in our leaderboard position and securing the third rank.

# References
1) Documentations from Scikit-Learn - https://scikit-learn.org/stable/index.html
2) Oyelade J, Isewon I, Oladipupo F, et al. Clustering Algorithms: Their Application to Gene Expression Data. Bioinform Biol Insights. 2016;10:237-253. Published 2016 Nov 30. doi:10.4137/BBI.S38316.





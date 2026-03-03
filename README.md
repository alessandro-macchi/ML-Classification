# Machine Learning Project
# Wine Quality Prediction: A Machine Learning Analysis

## Author
**Alessandro Macchi** - github.com/alessandro-macchi

## Overview
The traditional art of winemaking meets the mathematical precision of data science in this analytical project. At its core, this repository investigates the effectiveness of various machine learning approaches for binary wine quality classification. By analyzing eleven distinct physicochemical properties—ranging from fixed acidity and alcohol content to complex sulfur compounds—this framework attempts to predict whether a given wine will be perceived as low or high quality. 

A central thesis of this research is evaluating the necessity of algorithmic complexity: do the intricate, non-linear mappings of advanced kernel methods provide a justified performance enhancement over the interpretable simplicity of baseline linear models?

## Dataset

The Wine Quality dataset is sourced from the UCI Machine Learning Repository and includes physicochemical tests of various wines, both red and white. The goal is to predict wine quality as a binary classification problem.

Dataset link: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## The Data Ecosystem and Feature Engineering
To expose the true underlying signals within the physicochemical data, the raw dataset undergoes a rigorous and highly intentional preprocessing pipeline. Initial exploratory data analysis revealed significant feature skewness and a diverse range of variable scales across the eleven attributes. 

To systematically address these anomalies and prepare the data for robust modeling, the following steps were implemented:
* **Mathematical Transformations**: Logarithmic transformations were applied to heavily skewed features to normalize their distributions.
* **Scale Normalization**: Strict standardization was enforced to ensure all variables operate within the same mathematical range, preventing features with larger magnitudes from dominating the learning process.
* **Algorithmic Balancing**: The initial dataset exhibited a mild but notable class imbalance, with 63% of the samples representing high-quality wines. To ensure our models did not develop a bias toward the majority class, the Synthetic Minority Oversampling Technique (SMOTE) was deployed to synthetically generate minority observations and achieve parity.
* **Chronological Integrity**: The data was systematically divided, preserving a strict 20% holdout test set to validate the final analysis and absolutely prevent any data leakage during the training phase.

## Modeling Architecture
Our analytical framework benchmarks a spectrum of predictive algorithms designed to capture both straightforward linear trends and the intricate, hidden complexities within the chemical data. The architecture implements four distinct models:

1. **Linear Logistic Regression**: A foundational probabilistic approach utilizing standard gradient descent.
2. **Linear Support Vector Machine (SVM)**: A robust margin-maximizing classifier optimized via the highly efficient Pegasos (Primal Estimated Sub-Gradient Solver) algorithm.
3. **Kernel Logistic Regression**: An advanced, non-linear probabilistic model scaled for efficiency using mini-batch stochastic gradient descent and subsampling of support vectors.
4. **Kernel SVM**: A complex, high-dimensional classifier utilizing the kernelized version of the Pegasos algorithm to capture deep feature interactions.

To guarantee that these models were optimally configured before final evaluation, exhaustive hyperparameter tuning was conducted across the training set utilizing rigorous 5-fold cross-validation.

## Performance Evaluation and Insights
Model accuracy and generalization were rigorously measured across a comprehensive suite of classification metrics, including overall accuracy, precision, recall, F1-score, and ROC-AUC. 

**The Verdict**: The experimental results decisively demonstrate that kernel methods significantly outperform their linear counterparts. This outcome confirms the presence of highly important, non-linear relationships within the physicochemical composition of wine that simple linear boundaries fail to capture. 

Key analytical takeaways include:
* **The Champion Model**: **Kernel Logistic Regression** emerged as the superior algorithm, achieving the best overall generalization and predictive performance across almost all evaluation metrics.
* **The Optimization Standout**: **Kernel SVM** demonstrated the fastest mathematical convergence and the lowest absolute training loss, though it yielded a slightly higher misclassification rate than Kernel Logistic Regression.
* **The Linear Shortfall**: Both linear models, and particularly the Linear SVM, exhibited persistent mild underfitting, proving inadequate for the dataset's complexity. 
* **Feature Challenges**: Deep misclassification analysis revealed that certain chemical properties—specifically chlorides, fixed acidity, and sulphates—consistently confused all models, suggesting these features contain high inherent noise.

Ultimately, this predictive power comes with a significant operational consideration. The analysis highlights a stark trade-off in computational efficiency: while kernel models deliver an approximate 6% reduction in misclassifications, they require more than ten times the computational training time compared to their linear equivalents. This delicate balance between algorithmic accuracy and computational efficiency remains a crucial consideration for real-world deployment.

---
layout: null
---

<div align="center">

<h1>PolyStock: Dimensionality Comparison of E-Markets</h1>
<h2>Project Proposal</h2>
<p><strong>Team 37</strong></p>

</div>

## 1 INTRODUCTION

The stock market and prediction markets like Polymarket are two significant markets where money changes hands, and almost anyone can participate in them. They share many similarities, as both the stock market and Polymarket rely on making future predictions to increase the chances of making strategic investments and bets, leading to greater profits for investors and bettors. The main difference is that the stock market tends to involve long-term investment in company shares, whereas Polymarket is mainly betting on specific events, each one being a short-term result. Also, the stock market has been around since 1602, first established in the United States in 1792, but Polymarket have only existed since 2020.

<figure>
	<img src="https://placehold.co/600x240?text=Example+Figure+Placeholder" alt="Example figure placeholder" width="600" />
	<figcaption><em>EXAMPLE FIGURE—asdfasdfadsf fadsfal dsfalsdkflasdjf ladf jlaskdjlaf jdlaskfj ladkjfladsjfla</em></figcaption>
</figure>

### 1.1 Literature Review

Therefore, it is very important and useful to be able to make accurate and efficient future predictions of the stock market and events on Polymarket. Machine learning algorithms can play a significant role in doing so. For the stock market, one study looked at “the effectiveness of algorithms like decision trees, random forests, support vector machines (SVM) with different kernels, and K-Means Clustering… [1].” It found that SVM using a Radial Base Function performed the best with the tradeoff of requiring significantly more time than the other algorithms, with an 88% prediction accuracy and runtime anywhere from about 50-2700% longer than the other algorithms. For Polymarket, one model used Reinforcement Learning with Verifiable Rewards on top of Large Language Models to capture both the data and other relevant information like news headlines, competing with the most current and advanced models and improving probabilistic calibration, leading to an estimated 10% return on investment in a Polymarket trading simulation [2]. It is hard to find sources comparing the two due to the novelty of this project, but studies have shown that prediction markets were more accurate than analysts in making predictions for company earnings, with prediction markets being 68% accurate for a week in advance and 77% accurate for a day in advance, but analysts were only 62% accurate [3]. This suggests that there is a strong connection between making predictions with both the stock market and Polymarket.

### 1.2 Dataset Description

The first dataset being used is a stock market dataset, containing historical daily prices for all tickers (each has its own chart) currently trading on NASDAQ. Its features include the date, open, high, low, close, adj close, and volume for each ticker. There is also a csv file containing the general information about each ticker.
There are two datasets being used for the Polymarket data, one called “Polymarket Prediction Markets” and the other called “Full market data from polymarket.” The first one contains deduplicated prediction market events and individual markets from Polymarket. It has a separate dataset for events, with features like event metadata, trading metrics, status flags, timestamps, category tags, competitiveness scores, and market counts per event, and markets, with features like market details, trading activity, order book data, event references, and reward structures. The second one contains raw market data collected from Polymarket Gamma API. Its features include the book, holder, price every 4 hours, and trade for each market.

### 1.3 Dataset Link

For all datasets, more details and the actual datasets themselves can be found at these links:
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets/data?select=polymarket_markets.csv
https://www.kaggle.com/datasets/sandeepkumarfromin/full-market-data-from-polymarket

## 2 PROBLEM DEFINITION
Stock prediction algorithms are a staple in the machine learning world. Recently, however, polymarkets where users can place bets on things ranging from elections to sports outcomes have exploded, reaching ~$10b in monthly trade volume within less than two years of launch (see Figure 1). Our high level goal is to build a system which can optimally make investment decisions across both the stock and poly markets.

<figure>
	<iframe width="100%" height="420" frameborder="0" src="https://www.theblock.co/data/decentralized-finance/prediction-markets-and-betting/polymarket-and-kalshi-volume-monthly/embed" title="Polymarket, Polymarket US and Kalshi Volume (Monthly)"></iframe>
	<figcaption><em>Figure 1: Polymarket, Polymarket US, and Kalshi monthly volume trends.</em></figcaption>
</figure>

### 2.1 Problem

Because of the relative infancy of polymarkets, there is a lot of volatility in prices, risk, and options. This is contrary to more mature equity markets. There also may be much more diverse features which play a role in these younger markets, like social sentiment and narrative polarization. Similar trends may exist in the stock market, but it is hard to gauge transferability.

### 2.2 Motivation

This motivates a project which 1. identifies if the dimensionality of stock/poly markets is meaningfully different and 2. frames both domains as comparable forecasting tasks. This could be useful to identify which signals are transferable across markets and which are market-specific. This can guide future systems and help quantify when to prioritize cross-market and specialized learning.

## 3 METHODS

### 3.1 Data Preprocessing

#### Missing Value Handling
Missing values in time series data (due to holidays or collection gaps) could be handled using `sklearn.impute.SimpleImputer` with median strategy as baseline, and `sklearn.impute.KNNImputer` (k=5) for preserving local structure. Time-series-specific interpolation (linear/spline) via `pandas` will also be evaluated.

#### Feature Scaling
To ensure fair comparison across markets with different scales, `sklearn.preprocessing.StandardScaler` applys Z-score normalization (mean=0, std=1). For datasets with outliers, `sklearn.preprocessing.RobustScaler` (based on median and IQR) provides stronger robustness.

#### Feature Engineering and Time Series Processing
Technical indicators could be computed using `pandas` and `numpy`: volatility (standard deviation of returns), volume trends (moving average slopes), and log returns. For time series, lagged features (1/3/7 periods), rolling statistics, and cyclical encodings (day-of-week, month) will be generated. `sklearn.preprocessing.PolynomialFeatures` will create interaction features to capture non-linear relationships.

#### Target Variable Construction
For stock prediction, continuous targets (log returns: log(Price_t+1/Price_t)) and discrete targets (three-class: up/down/flat based on ±1% threshold) could be constructed using `pandas` and `numpy`. For Polymarket, the target will be the difference between actual settlement results and current implied probability.

### 3.2 Unsupervised Learning Methods

#### Principal Component Analysis (PCA)
`sklearn.decomposition.PCA` could reduce dimensionality while maximizing variance retention. The number of components explaining 95% variance will quantify intrinsic dimensionality differences between stock and Polymarket markets. PCA-reduced features will also serve as inputs to supervised models.

#### K-Means Clustering and Gaussian Mixture Models (GMM)
`sklearn.cluster.KMeans` will identify market states by minimizing within-cluster sum of squares, with optimal K determined by silhouette score (`sklearn.metrics.silhouette_score`) or elbow method. `sklearn.mixture.GaussianMixture` will handle irregular cluster shapes using probabilistic clustering, with model selection via Bayesian Information Criterion (BIC).

#### Nonlinear Dimensionality Reduction and Visualization
A `tensorflow.keras` autoencoder could extract nonlinear representations of complex market structures. For 2D visualization, UMAP (`umap-learn`) and t-SNE (`sklearn.manifold.TSNE`) will project the data to highlight distribution differences between markets.

### 3.3 Supervised Learning Methods

#### Random Forest and XGBoost
`sklearn.ensemble.RandomForestClassifier/Regressor` could serve as the ensemble method, with hyperparameters (n_estimators, max_depth, min_samples_split) optimized via `sklearn.model_selection.GridSearchCV`. XGBoost (`xgboost` library) provides gradient boosting with parallel computation and built-in regularization to prevent overfitting.

#### Support Vector Machines (SVM)
`sklearn.svm.SVC` and `sklearn.svm.SVR` implement classification and regression with RBF kernels for capturing nonlinear relationships. Parameters C and gamma will be tuned via grid search. PCA preprocessing will be applied for computational efficiency on large datasets.

#### Linear and Regularized Regression
`sklearn.linear_model.LinearRegression` could provide the baseline. Ridge (`Ridge`, L2 regularization), Lasso (`Lasso`, L1 feature selection), and Elastic Net (`ElasticNet`, combined L1/L2) will prevent overfitting. Regularization strength alpha will be selected via cross-validation.

#### Long Short-Term Memory (LSTM)
LSTM networks implemented via `tensorflow.keras.layers.LSTM` or `torch.nn.LSTM` models long-term dependencies in time series data. Unlike traditional ML with fixed lag features, LSTM learns adaptive temporal patterns through gating mechanisms. Architecture: LSTM layers (50-100 units) followed by dense output layers. Dropout (0.2-0.3) prevents overfitting. Early stopping halts training when validation loss plateaus.

## 4 RESULTS AND DISCUSSION

### 4.1 Project Goals (quantitative)
Our primary goal is to determine (in a quantitative manner) if prediction markets such as Polymarket exhibit lower intrinsic dimensionality than your tranditional equity markets (NYSE, NASDAQ), and to use these findings to build optimal forecasting models across both domains.

Quantitative metrics that we will use to evaluate our methodology:
- Explained Variance Ratio (explained_variance_)
Used during PCA to quantify the intrinsic dimensionality of each market. We will measure the proportion of the dataset's variance captured by each principal component.
Comparing how many components are needed to reach a 95% explained variance threshold will serve as our primary comparative measure.
- Mean Squared Error (mean_squared_error)
MSE is our primary risk metric for evaluating the predictive accuracy of our Random Forest, XGBoost, and LSTM models, representing the expected value of the squared error.
- R^2 Score (r2_score)
R_2 score is our coefficient of determination, being used to evaluate the goodness-of-fit for our predictive models. It is necessary to measure the proportion of variance in future market prices, explained by our lagged features.
- Silhouette Score (silhouette_score)
Silhouette Score is used to evaluate our K-Means and GMM models to make sure that clusters are cohesive and well-seperated from each other (clusters being discovered market regimes).

### 4.2 Expected Results (quantitative)

## 5 REFERENCES

[1]		A. Chakravorty and N. Elsayed, “A Comparative Study of Machine Learning Algorithms for Stock Price Prediction Using 		Insider Trading Data,” Arxiv.org, 2025. [Online]. Available: https://arxiv.org/html/2502.08728v1. [Accessed February 		23, 2026].

[2]		B. Turtel, D. Franklin, K. Skotheim, L. Hewitt, and P. Schoenegger, “Outcome-based Reinforcement Learning to Predict 		the Future,” Arvix.org, 2025. [Online]. Available: https://arxiv.org/pdf/2505.17989. [Accessed February 23, 2026].

[3]		G. McCubbing, “Can Polymarket-style prediction markets beat analysts this earnings season? Polymarket thinks so,” 			Australian Financial Review, February 4, 2026. [Online]. Available: https://www.afr.com/markets/equity-markets/can-			prediction-markets-beat-analysts-this-earnings-season-20260202-p5nyul [Accessed February 23, 2026].

EXAMPLE: Sorokin, I., Puget J. F. (2025) NVARC solution to ARC-AGI-2 2025. Google Drive. [https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view](https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view)

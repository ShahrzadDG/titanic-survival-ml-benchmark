# Titanic Survival Prediction: Machine Learning Benchmark

This project investigates passenger survival prediction using the Kaggle Titanic dataset. The objective is to develop and evaluate binary classification models through a machine learning workflow, from exploratory data analysis and feature engineering to hyperparameter optimization and ensemble learning.

The analysis compares classical machine learning algorithms, including Logistic Regression, k-Nearest Neighbor (KNN), Decision Tree, Random Forest, Extra Trees, Support Vector Machines, Gradient Boosting, XGBoost, LightGBM, and CatBoost. It also evaluates a multilayer perceptron (MLP) as a neural network benchmark.

Model evaluation includes stratified cross-validation, GridSearchCV, Optuna, nested cross-validation, and analysis of predictive performance and feature importance.

**Key objectives:**
- Explore relationships between passenger characteristics and survival.
- Develop reproducible feature engineering and preprocessing pipelines.
- Compare classification algorithms and optimization strategies.
- Evaluate generalization and potential overfitting.
- Investigate whether ensemble learning or neural networks improve predictive performance.
- Generate and evaluate submissions on the Kaggle public leaderboard.

The emphasis is on methodological comparison, reproducibility, and careful interpretation of model performance.

** Kaggle Leaderboard Results **

Two models were submitted to the Kaggle Titanic competition.

| Model | Public leaderboard accuracy |
|---|---:|
| Soft-Voting Ensemble (Random Forest + LightGBM + CatBoost) | **0.7751** |
| Tuned Multilayer Perceptron (MLP) | 0.7727 |

The soft-voting ensemble achieved a slightly higher public leaderboard accuracy than the tuned MLP, with a difference of 0.24 percentage points.

The models produced different predictions for some test passengers, indicating that their learned decision boundaries were not identical. However, these disagreements cannot be evaluated at the individual passenger because the competition test labels are hidden.

The main findings are:

- Passenger sex, ticket class, and age showed important associations with survival.
- Feature engineering improved the logistic regression baseline in cross-validation.
- Tree-based ensemble methods achieved stronger validation performance than the tested neural network configurations.
- Increasing neural network complexity did not automatically improve predictive performance on this relatively small dataset.

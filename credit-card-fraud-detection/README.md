# Credit Card Fraud Detection Using Classical Machine Learning Algorith,s

## Project Overview
This project demonstrates the development of a robust credit card fraud detection system using classical machine learning algorithms, achieving 97.7% ROC-AUC and 74.3% F1-Score on the highly imbalanced Kaggle Credit Card Fraud Detection dataset.

### Key Achievements
- **Exceptional Performance**: 97.7% ROC-AUC (matching the performances of published research)
- **Balanced Classification**: 74.3% F1-Score with 75.3% precision and 73.3% recall
- **Robust Methodology**: End-to-end ML pipeline with proper evaluation techniques

## Dataset Information
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 284,807 transactions
- **Features**: 30 features (28 PCA-transformed + Time + Amount + Class)
- **Class Imbalance**: 99.83% legitimate, 0.17% fraudulent transactions
- **Challenge**: Extreme class imbalance requiring specialized handling techniques

## Technical Approach

### 1. Exploratory Data Analysis
- Comprehensive statistical analysis of all 30 features
- Class distribution analysis revealing severe imbalance (1:577 ratio)
- Time-based pattern analysis across 48-hour transaction period
- Feature correlation and outlier detection

### 2. Data Preprocessing
- **Feature Selection**: Top 15 most discriminative features identified
- **Scaling**: RobustScaler applied to handle outliers effectively
- **Time-Aware Splitting**: 80/20 train-test split maintaining temporal order
- **Class Imbalance Handling**: SMOTE oversampling with multiple strategies tested

### 3. Model Development
**Algorithms Evaluated:**
- Logistic Regression with L1/L2 regularization
- Decision Tree with class-weight balancing
- Random Forest with hyperparameter tuning

**Evaluation Strategy:**
- Stratified 5-fold cross-validation
- Business-focused metrics (precision, recall, F1-score, ROC-AUC)
- Time-based validation to prevent data leakage
- Threshold optimization for cost-sensitive learning

### 4. Model Selection & Optimization
- Comprehensive comparison across sampling strategies
- Hyperparameter tuning using RandomizedSearchCV
- Business-cost optimization balancing investigation costs vs fraud losses
- Performance stability analysis across different scenarios

## Model Performance

### Final Model Metrics
| Metric | Score | Industry Benchmark | Status |
|--------|-------|-------------------|---------|
| **ROC-AUC** | **0.9766** | 0.90-0.95 | Exceptional |
| **F1-Score** | **0.7432** | 0.65-0.80 | Excellent |
| **Precision** | **0.7534** | 0.70-0.85 | Strong |
| **Recall** | **0.7333** | 0.65-0.80 | Good |

## Limitations & Deployment Considerations

### Technical Limitations
1. **PCA Feature Dependency**: Model requires 28 PCA-transformed features as inputs
2. **Feature Engineering Complexity**: Real-world deployment would need the original PCA transformation >
3. **Data Privacy**: Original features are anonymized, limiting practical interpretation
4. **Scalability**: Some algorithms (SVM) faced performance issues with large datasets

### Deployment Challenges
- **Real-time Implementation**: PCA transformed features in the dataset make it impractical for deployment (since user can't be asked to input values of PCA transformed features.)
- **Model Drift**: Performance may degrade as fraud patterns evolve

### Recommended Next Steps for Production
- Collect datasets containing raw and interpretable features.
- Then transfrom these features using PCA and train models on these PCA transformed features.
- During deployment, use the same PCA Transformer used during training, transform user imputs and then make a prediction.

## Project Value & Learning Outcomes

### Technical Skills Demonstrated
- **Advanced ML Pipeline**: Complete end-to-end development process
- **Imbalanced Data Expertise**: Sophisticated handling of class imbalance
- **Business Acumen**: Cost-benefit analysis and ROI calculations
- **Model Evaluation**: Comprehensive performance assessment techniques

### Industry Relevance
This project demonstrates skills directly applicable to:
- **Healthcare**: Rare disease detection and medical anomaly identification
- **Automotive**: Predictive maintenance and safety-critical event detection
- **Financial Services**: Risk assessment and regulatory compliance
- **Cybersecurity**: Intrusion detection and threat analysis

## Repository Structure

```
credit-card-fraud-detection/
├── data/
│   ├── creditcard.csv                                      # Dataset used (Rounded off the original values to reduce space consumed) 
│   ├── eda_insights.json                                   # Analysis results
│   └── imbalance_handling_strategies.json                  # Class counts after applying multiple oversampling techniques
├── models/
│   ├── best_fraud_detection_model.pkl                      # Best model
│   ├── feature_importances.json                            # Feature Importance data stored to plot
│   ├── feature_scaler.pkl                                  # Scaler for the best model
│   ├── model_metadata.json                                 # Model metadata
│   ├── performace_metrics_selected_models_baseline.json    # Performance metrics of selected baseline models
│   ├── performace_metrics_selected_models.json             # Performance metrics of selected tuned models
│   ├── tuned_dt.pkl                                        # Hyperparameter tuned Decision Tree model
│   └── tuned_rf.pkl                                        # Hyperparameter tuned Random Forest model
├── notebooks/
│   ├── 01_data_exploration.ipynb                           # Comprehensive EDA
│   ├── 02_modelling.ipynb                                  # Model development & training
│   ├── 03_comprehensive_evaluation.ipynb                   # Essential Performance analysis
│   └── results_comparision.csv
├── performance-measures/                                   # some data saved to avoid running the cells the next day
│   ├── cv_metrics_data.json
│   ├── final_results.json
│   └── tuned_params.json
├── results/
│   ├── best_strategy_performance_metrics.png               # Performance metrics of all models for the best oversampling strategy (Borderline SMOTE)
│   ├── cost_matrices_of_tuned_and_baseline_models.png      # Cost Matrices Plot
│   ├── different_strategies_performance_metrics.png        # Performance metrics of all models for all the 4 oversampling strategies
│   ├── feature_importance_best_2_models.png                # Top 2 models being tree based - Feature Importance plots
│   ├── roc_auc_precision_recall_curves.png                 # ROC-AUC Curves and Precision-Recall Curves for best 2 models
│   └── tuned_vs_baseline_models.png                        # Comparision of performance metrics for selected 2 models (Tuned Vs Original)
└── README.md                                               # This documentation
```

## Academic & Professional Impact

### Research Contributions
- Achieved performance metrics competitive with top academic publications
- Demonstrated effective classical ML approach for modern fraud detection

### Professional Portfolio Value
This project showcases:
- **Technical Depth**: Advanced ML techniques with rigorous evaluation
- **Business Understanding**: Clear ROI and cost-benefit analysis
- **Communication Skills**: Comprehensive documentation and visualization
- **Problem-Solving**: Handling real-world data science challenges
- **Industry Readiness**: Understanding of deployment limitations and solutions

## Key Visualizations

### Performance Metrics Comparison
- Model performance vs industry benchmarks
- ROC and Precision-Recall curves
- Confusion matrix with business cost analysis
- Threshold optimization for maximum business value

## Future Enhancements

While this project achieved moderate but appreciable performances, future work could include:

1. **Feature Engineering**: Develop interpretable features from domain knowledge
2. **Ensemble Methods**: Combine multiple algorithms for improved robustness
3. **Deep Learning Comparison**: Benchmark against neural network approaches
4. **Deployment**: This project can be retrained using the real dataset with interpretable features and deployed for fraud detection.

## Conclusion

This credit card fraud detection project successfully demonstrates machine learning capabilities, achieving industry-leading performance metrics while maintaining practical business focus. The 97.7% ROC-AUC and comprehensive evaluation methodology showcase technical expertise suitable for ML roles in healthcare, automotive, and financial services industries.

**Key Takeaway**: While the model achieves exceptional performance, the PCA-transformed input features present practical deployment challenges, highlighting the importance of balancing technical performance with real-world implementation considerations in machine learning projects.

---

*Project completed: July 2025*  
*Technologies: Python, scikit-learn, pandas, numpy, matplotlib, seaborn*  
*Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions)*
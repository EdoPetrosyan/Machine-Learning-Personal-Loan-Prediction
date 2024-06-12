# Personal loan prediction


The project aims to build a predictive model to determine whether an individual is likely to be approved for a personal loan based on various features provided in the dataset.


## Dependencies


    Python 3.x
    NumPy
    Pandas
    Scikit-learn
    Matplotlib
    Seaborn
    XGBoost
    Random Forest

## Dataset
Dataset is taken from Kaggle webpage.
The link to the dataset:
https://www.kaggle.com/code/alirezachahardoli/bank-personal-loan-modelling/notebook

The dataset consists of 5000 rows and 14 columns, with the "personal_loan" column serving as the target variable.

The features are:

    • id : Customer ID

    • age : Customer's age in completed years

    • experience : years of professional experience

    • income : Annual income of the customer in thousands

    • zip_code : Home Address ZIP code.

    • family : Family size of the customer

    • ccavg : Avg. spending on credit cards per month

    • education : Education Level. Undergrad:1 Graduate:2 Advanced/Professional:3

    • mortgage : Value of house mortgage if any. • personal_loan : Did this customer accept the personal loan offered in the last campaign? Accepted:1 Denied:0

    • securities_account : Does the customer have a securities account with the bank? Yes:1, No:0

    • cd_account : Does the customer have a certificate of deposit (CD) account with the bank? Yes:1 No:0

    • online : Does the customer use internet banking facilities? Yes:1 No:0

    • creditcard : Does the customer use a credit card issued by Universal Bank? Yes:1 No:0



## Models Employed:
1. **Logistic Regression**:
   - Logistic regression is chosen for its interpretability and ability to model the probability of loan approval based on independent variables.
   - It provides insights into the relationship between features and the likelihood of loan acceptance.

2. **Support Vector Machine (SVM)**:
   - SVM is selected for its effectiveness in identifying complex decision boundaries and handling high-dimensional data.
   - It's particularly useful in scenarios where clear margins of separation between loan approval classes are essential.

3. **k-Nearest Neighbors (KNN)**:
   - KNN captures local patterns in the data and makes predictions based on the majority class of neighboring instances.
   - Its simplicity and adaptability to varying data distributions make it a valuable addition to the modeling approach.

4. **Bagging (Bootstrap Aggregating)**:
   - Bagging improves model stability and generalization by combining multiple models trained on different subsets of the training data.
   - It mitigates overfitting and variance issues, enhancing the overall performance of the predictive system.

5. **Gradient Boosting (XGBoost)**:
   - XGBoost is employed to handle complex relationships in the data and optimize model performance through gradient boosting.
   - It's known for its scalability, efficiency, and ability to handle large datasets with high-dimensional features.

6. **Random Forest**:
   - Random Forest is utilized for its ensemble approach, combining multiple decision trees to make predictions.
   - It's robust against overfitting and noise, offering improved accuracy and generalization compared to individual decision trees.


## Training

- Models were trained using the provided dataset.
- Data preprocessing techniques such as scaling and encoding were applied before training.
- Hyperparameters were tuned using techniques K-fold cross validation, grid search and random search

## Evaluation


- Models were evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC AUC.
- Performance metrics were compared between initial and tuned models.

You can find detailed Evaluation in the notebook. 

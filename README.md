This project focuses on multivariate time series forecasting using LSTM, ARIMA and SHAP explainability. It includes dataset generation, preprocessing, model training, evaluation and interpretability.

Installation
Install all required packages using the command below:
pip install shap pandas scikit-learn statsmodels tensorflow matplotlib

Dataset Generation
A synthetic dataset is created with three columns: feature1, feature2 and target.
The values include patterns, trends and noise to simulate real time series data.

Preprocessing
The data is scaled using MinMaxScaler.
A sliding window of size 30 is created.
Inputs are the past 30 values of both features.
Output is the next value of the target column.
The dataset is split into train and test sets without shuffling.

LSTM Model
The model has one LSTM layer with 64 units and one Dense output layer.
The model is trained using the Adam optimizer, MAE loss and MSE metric for 20 epochs.

Model Evaluation
Evaluation is done using MAE and RMSE on the test set.

ARIMA Baseline
An ARIMA(5,1,0) model is trained using only the target column.
Its predictions are compared with the LSTM predictions.

SHAP Explainability
SHAP KernelExplainer is used to interpret the LSTM model.
Inputs are flattened for SHAP and reshaped back inside a wrapper function.
A background sample is chosen from training data.
SHAP values are computed and waterfall plots are generated.
Time-lagged feature names are used to show the effect of each past time step.

How to Run
Install dependencies.
Run the notebook cell by cell.
Compare LSTM and ARIMA metrics.
View SHAP plots to understand model decisions.

Future Improvements
Add hyperparameter tuning.
Try GRU or Bidirectional LSTM.
Use real datasets.
Deploy the model as an API.


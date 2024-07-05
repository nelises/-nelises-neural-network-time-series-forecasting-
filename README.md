# -nelises-neural-network-time-series-forecasting-
Examining Time Series Forecasting Approaches for Predicting Canadian Employment Trends

# README for Neural Network Time Series Forecasting Project

## Overview
This repository contains the project "Examining Time Series Forecasting Approaches for Predicting Canadian Employment Trends", a part of the CIND 820: Big Data Analytics Project at Toronto Metropolitan University. The project aims to predict Canadian employment trends by leveraging advanced predictive analytics and big data methodologies. The study employs ARIMA, LSTM, and the novel TimesFM model to forecast unemployment rates and employment wages.

## Abstract
The rising unemployment rate in Canada is a significant concern for the nation's economic stability. This project seeks to employ advanced predictive analytics and big data methodologies to forecast Canadian employment trends. The project leverages statistical and deep learning models, including ARIMA, LSTM networks, and the innovative TimesFM model, to predict unemployment rates and employment wages. The models will be trained using comprehensive datasets from Statistics Canada, encompassing unemployment and wage data spanning from 1997 to 2023. The implementation of this project will be conducted using Python, facilitated by the Visual Studio Code (VS Code) editor, and the project’s code repository will be stored and managed through GitHub.

## Repository Structure
The repository contains the following main components:

1. **Data Collection**
   - **Source:** Statistics Canada
     - Unemployment Data (1997-2023)
     - Wage Data (1997-2023)
   - **Variables:** Canadian province, occupation or industry classification, demographic
         information (age group, sex) 
    ***Wage:*** The average hourly rate for full-time employees
     ***Unemployment:*** The unemployment rate (%)

2. **Data Preprocessing**
   - **Tools:** Pandas, NumPy, Sklearn
   - **Steps:**
     - Data Cleaning
     - Data Transformation
     - Handling Missing Values

3. **Model Selection and Development**
   - **Models:**
     - ARIMA (Autoregressive Integrated Moving Average)
     - LSTM (Long Short-Term Memory) Networks
     - TimesFM (Large Language Model for Time-Series Forecasting)

4. **Model Implementation**
   - **Tools:** Python, Visual Studio Code (VS Code), GitHub
   - **Run Models:**
     - Statsmodels (ARIMA)
     - TensorFlow and Keras (LSTM)
     - Timesfm package (LLM)

5. **Model Training and Testing**
   - **Training Data:** 1997-2020
   - **Testing Data:** 2021-2023
   - **Validation:** Ensure robust model validation

6. **Cross-Validation**
   - TimeSeriesSplit cross-validation is used to validate model performance over multiple training and validation splits to ensure robustness.

7. **Analysis and Interpretation**
   - Evaluate model performance
   - Compare model predictions
   - Visualize results

8. **Present Results**
   - Provide insights on employment trends
   - Suggest preventative measures for policymakers

## Project Stages

1. **Initial Setup and Data Collection**
   - Download and explore datasets from Statistics Canada.
   - Preliminary analysis and visualization of data to understand trends and patterns.

2. **Data Preprocessing**
   - Cleaning and transforming the data to ensure quality and consistency.
   - Handling missing values and performing feature engineering to enhance the dataset.

3. **Model Development**
   - Selecting appropriate models (ARIMA, LSTM, TimesFM) for the forecasting task.
   - Developing and tuning models using historical data.

4. **Model Implementation and Cross-Validation**
   - Implementing models in Python using relevant libraries (Statsmodels, TensorFlow, Timesfm).
   - Performing cross-validation to validate model performance.

5. **Model Training and Testing**
   - Training models on the dataset from 1997-2020.
   - Testing models on the dataset from 2021-2023 to evaluate their predictive performance.

6. **Analysis and Visualization**
   - Analyzing model outputs and comparing predictions.
   - Visualizing results to interpret model effectiveness and accuracy.

7. **Reporting and Documentation**
   - Documenting the methodology, results, and insights gained from the analysis.
   - Preparing a comprehensive report to present findings and recommendations.

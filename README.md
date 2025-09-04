# Weather-data-classification
The weather data classification is prediction application which is designed to classify the humidity level. It uses the machine learning, specifically a decision Tree Classifier, predicts the humidity and 3pm 

## 1. Data Loading and Preprocessing
- The project starts by loading a CSV file (daily_weather.csv) which contains daily weather data.
- It then handles missing values in the data by forward filling (fillna(method='ffill')).
- A new feature humidity_class is created based on the relative_humidity_3pm column, with values categorized as 1 for humidity greater than 25% (high) and 0 for humidity less than or equal to 25% (low).

## 2. Model Training
- The dataset is split into features (x) and target (y), with y being the newly created humidity_class.
- The data is then divided into training and test sets using train_test_split.
- A Decision Tree Classifier is trained on the training data (x_train, y_train), and the model is saved as model.pkl using joblib.

## 3. Model Evaluation

After training, the model is evaluated on the test set (x_test and y_test), and its accuracy is displayed via Streamlit (st.write).

The project also generates and displays a confusion matrix to provide insight into the performance of the model by showing the actual vs predicted classifications.

4. User Interface (Streamlit)

- The app provides an interactive interface using Streamlit where users can input different weather parameters:
   - Air Pressure at 9 AM
   - Air Temperature at 9 AM
   - Average Wind Direction at 9 AM
   - Average Wind Speed at 9 AM
   - Maximum Wind Direction at 9 AM
   - Maximum Wind Speed at 9 AM
   - Rain Accumulation at 9 AM
   - Rain Duration at 9 AM
   - Air Temperature at 3 PM

- Once the user enters these parameters, they can click a button to get a prediction of whether the humidity at 3 PM is above or below 25%.

5. Prediction and Output

Upon clicking the "Predict Humidity Class" button, the model predicts whether the humidity is high (greater than 25%) or low (less than or equal to 25%) based on the input parameters.

The prediction result is displayed on the Streamlit interface, either confirming a high or low humidity prediction.

6. Visualization

A scatter plot is shown comparing the actual vs predicted humidity class values.

A confusion matrix is also presented, which shows the model’s performance across all test samples (true positives, true negatives, false positives, and false negatives).

Technologies and Libraries Used:

Pandas: For data handling and preprocessing.

Scikit-learn: For machine learning (train-test split, Decision Tree Classifier, and model evaluation).

Streamlit: For creating the interactive web interface.

Joblib: To save and load the trained model.

Matplotlib: For plotting the actual vs predicted values and the confusion matrix.

Conclusion:

This project is a simple but effective way to predict the humidity of a given day based on weather data, making it useful for understanding weather patterns or integrating into applications that require this kind of weather classification.

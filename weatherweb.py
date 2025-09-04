import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://c1.wallpaperflare.com/preview/830/920/873/night-stars-sky-dark.jpg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load and clean data
df = pd.read_csv("daily_weather.csv")
df.fillna(method='ffill', inplace=True)
df['humidity_class'] = df['relative_humidity_3pm'].apply(lambda x: 1 if x > 25 else 0)

# Prepare features and target
x = df.drop(columns=["relative_humidity_3pm", "humidity_class", "number", "Unnamed: 11"], errors='ignore')
y = df["humidity_class"]

# Train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

# Save model
joblib.dump(dt_model, "model.pkl")
print("✅ Model saved as model.pkl")

# Load model
dt_model = joblib.load("model.pkl")

# Streamlit UI
st.title("Weather Humidity Classifier")
st.write("Enter weather parameters to predict humidity")

# Input fields
air_pressure_9am = st.number_input("Air Pressure at 9AM", value=920.0)
air_temp_9am = st.number_input("Air Temperature at 9AM", value=65.0)
avg_wind_direction_9am = st.number_input("Avg Wind Direction at 9AM", value=190.0)
avg_wind_speed_9am = st.number_input("Avg Wind Speed at 9AM", value=4.0)
max_wind_direction_9am = st.number_input("Max Wind Direction at 9AM", value=205.0)
max_wind_speed_9am = st.number_input("Max Wind Speed at 9AM", value=5.0)
rain_accumulation_9am = st.number_input("Rain Accumulation at 9AM", value=0.0)
rain_duration_9am = st.number_input("Rain Duration at 9AM", value=0.0)
air_temp_3pm = st.number_input("Air Temperature at 3PM", value=26.0)

# Prediction
if st.button("Predict Humidity Class"):
    new_sample = [[air_pressure_9am, air_temp_9am, avg_wind_direction_9am, avg_wind_speed_9am,
                   max_wind_direction_9am, max_wind_speed_9am, rain_accumulation_9am,
                   rain_duration_9am, air_temp_3pm]]
    prediction = dt_model.predict(new_sample)
    if (prediction[0]==1):
         st.success(f"Predicted Humidity is greater than 25")
    else:
          st.success(f"Predicted Humidity is less than 25")
   
# Show model accuracy
accuracy = dt_model.score(x_test, y_test)
st.subheader("Model Accuracy")
st.write(f"Accuracy on test data: **{accuracy * 100:.2f}%**")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Predict on test set
y_pred = dt_model.predict(x_test)

# Plot actual vs predicted
st.subheader("Actual vs Predicted Humidity Class")

fig, ax = plt.subplots()
ax.scatter(range(len(y_test)), y_test, label='Actual', color='Red', alpha=0.6)
ax.scatter(range(len(y_pred)), y_pred, label='Predicted', color='orange', alpha=0.6)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Humidity Class")
ax.legend()
st.pyplot(fig)


st.subheader("Confusion Matrix")

# Create confusion matrix DataFrame
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(
    cm,
    columns=["Predicted less than 25", "Predicted Greater than 25"],
    index=["Actual less than 25", "Actual Greater than 25"]
)

# Convert DataFrame to HTML with custom styling
html_table = df_cm.to_html(classes="styled-table")

# Inject custom CSS and HTML
st.markdown("""
    <style>
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        background-color: rgb(11, 11, 66);
        color: white;
        font-family: Arial, sans-serif;
        border-radius: 5px;
        overflow: hidden;
    }
    .styled-table th, .styled-table td {
        border: 1px solid white;
        
    }
    .styled-table th {
        min-width: 200px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(html_table, unsafe_allow_html=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import joblib

# # 🎨 Page setup
# st.set_page_config(page_title="Mobile Addiction Predictor", page_icon="📱", layout="wide")
# st.title("📱 Mobile Addiction Level Prediction")
# st.markdown("Fill in the details below to estimate the **Addiction Level**.")

# # 🔹 Load your trained model
# # Make sure model.pkl was saved with joblib.dump() in the same sklearn version
# @st.cache_resource
# def load_model():
#     return joblib.load("dt_model.pkl")

# model = load_model()

# try:
#     training_columns = joblib.load("columns.pkl")
# except:
#     training_columns = None

# # model = joblib.load("dt_model.pkl")

# # If you saved training columns during preprocessing, load them here
# # Example: training_columns = joblib.load("columns.pkl")
# training_columns = None  # replace with actual list if you saved it

# # 🧑‍💻 Input form
# with st.form("user_inputs"):
#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.slider("Age", 10, 60, 18)
#         gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#         daily_usage = st.slider("Daily Usage Hours", 0.0, 12.0, 4.0)
#         sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
#         intellectual_perf = st.slider("Intellectual Performance", 0, 100, 75)
#         social_interactions = st.slider("Social Interactions", 0, 20, 5)
#         exercise_hours = st.slider("Exercise Hours", 0.0, 5.0, 1.0)
#         anxiety_level = st.slider("Anxiety Level", 1, 10, 5)
#         depression_level = st.slider("Depression Level", 1, 10, 5)

#     with col2:
#         self_esteem = st.slider("Self Esteem", 1, 10, 5)
#         screen_time_bed = st.slider("Screen Time Before Bed (hrs)", 0.0, 5.0, 1.0)
#         phone_checks = st.slider("Phone Checks Per Day", 0, 200, 50)
#         apps_used = st.slider("Apps Used Daily", 1, 50, 10)
#         time_social = st.slider("Time on Social Media (hrs)", 0.0, 10.0, 2.0)
#         time_gaming = st.slider("Time on Gaming (hrs)", 0.0, 10.0, 1.0)
#         time_education = st.slider("Time on Education (hrs)", 0.0, 10.0, 2.0)
#         phone_usage_purpose = st.selectbox("Phone Usage Purpose", ["Browsing", "Gaming", "Education", "Social Media", "Other"])
#         family_comm = st.slider("Family Communication (per day)", 0, 10, 3)
#         weekend_usage = st.slider("Weekend Usage Hours", 0.0, 15.0, 5.0)

#     submitted = st.form_submit_button("🔍 Predict Addiction Level")

# # 📊 Collect inputs into DataFrame
# if submitted:
#     input_data = pd.DataFrame({
#         "Age": [age],
#         "Gender": [gender],
#         "Daily_Usage_Hours": [daily_usage],
#         "Sleep_Hours": [sleep_hours],
#         "Interllectual_Performance": [intellectual_perf],
#         "Social_Interactions": [social_interactions],
#         "Exercise_Hours": [exercise_hours],
#         "Anxiety_Level": [anxiety_level],
#         "Depression_Level": [depression_level],
#         "Self_Esteem": [self_esteem],
#         "Screen_Time_Before_Bed": [screen_time_bed],
#         "Phone_Checks_Per_Day": [phone_checks],
#         "Apps_Used_Daily": [apps_used],
#         "Time_on_Social_Media": [time_social],
#         "Time_on_Gaming": [time_gaming],
#         "Time_on_Education": [time_education],
#         "Phone_Usage_Purpose": [phone_usage_purpose],
#         "Family_Communication": [family_comm],
#         "Weekend_Usage_Hours": [weekend_usage]
#     })

#     st.subheader("📋 Your Input Data")
#     st.write(input_data)

#     # ⚙️ Preprocess categorical variables (must match training preprocessing!)
#     input_encoded = pd.get_dummies(input_data)
    
#     try:
#         training_columns = joblib.load("columns.pkl")
#     except:
#         training_columns = None

#     # input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
#     # # Align with training columns if available
#     # # Align with training columns if available
#     # if training_columns is not None and len(training_columns) > 0:
#     #     input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)


#     # # 🚀 Prediction
#     # prediction = model.predict(input_encoded)[0]
#     # st.success(f"📱 Predicted Addiction Level: **{prediction:.2f}**")



#     try:
#         if training_columns is not None:
#             input_encoded = pd.get_dummies(input_data)
#             input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
#             prediction = model.predict(input_encoded)[0]
#         else:
#             prediction = model.predict(input_data)[0]

#         st.success(f"📱 Predicted Addiction Level: **{prediction:.2f}**")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import joblib

# 🎨 Page setup
st.set_page_config(page_title="Mobile Addiction Predictor", page_icon="📱", layout="wide")
st.title("📱 Mobile Addiction Level Prediction")
st.markdown("Fill in the details below to estimate the **Addiction Level**.")

# 🔹 Load your trained model (cached so it loads only once)
@st.cache_resource
def load_model():
    return joblib.load("dt_model (1).pkl")   # must be saved with joblib.dump

model = load_model()

# Try to load training columns (only needed if you used pd.get_dummies during training)
try:
    training_columns = joblib.load("columns (1).pkl")
except:
    training_columns = None

# 🧑‍💻 Input form
with st.form("user_inputs"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 10, 60, 18)
        gender = st.selectbox("Gender", ["Male", "Female"])  # restrict to categories used in training
        daily_usage = st.slider("Daily Usage Hours", 0.0, 12.0, 4.0)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        intellectual_perf = st.slider("Intellectual Performance", 0, 100, 75)
        social_interactions = st.slider("Social Interactions", 0, 20, 5)
        exercise_hours = st.slider("Exercise Hours", 0.0, 5.0, 1.0)
        anxiety_level = st.slider("Anxiety Level", 1, 10, 5)
        depression_level = st.slider("Depression Level", 1, 10, 5)

    with col2:
        self_esteem = st.slider("Self Esteem", 1, 10, 5)
        screen_time_bed = st.slider("Screen Time Before Bed (hrs)", 0.0, 5.0, 1.0)
        phone_checks = st.slider("Phone Checks Per Day", 0, 200, 50)
        apps_used = st.slider("Apps Used Daily", 1, 50, 10)
        time_social = st.slider("Time on Social Media (hrs)", 0.0, 10.0, 2.0)
        time_gaming = st.slider("Time on Gaming (hrs)", 0.0, 10.0, 1.0)
        time_education = st.slider("Time on Education (hrs)", 0.0, 10.0, 2.0)
        phone_usage_purpose = st.selectbox("Phone Usage Purpose", ["Browsing", "Gaming", "Education", "Social Media"])
        family_comm = st.slider("Family Communication (per day)", 0, 10, 3)
        weekend_usage = st.slider("Weekend Usage Hours", 0.0, 15.0, 5.0)

    submitted = st.form_submit_button("🔍 Predict Addiction Level")

# 📊 Collect inputs into DataFrame
if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Daily_Usage_Hours": [daily_usage],
        "Sleep_Hours": [sleep_hours],
        "Interllectual_Performance": [intellectual_perf],
        "Social_Interactions": [social_interactions],
        "Exercise_Hours": [exercise_hours],
        "Anxiety_Level": [anxiety_level],
        "Depression_Level": [depression_level],
        "Self_Esteem": [self_esteem],
        "Screen_Time_Before_Bed": [screen_time_bed],
        "Phone_Checks_Per_Day": [phone_checks],
        "Apps_Used_Daily": [apps_used],
        "Time_on_Social_Media": [time_social],
        "Time_on_Gaming": [time_gaming],
        "Time_on_Education": [time_education],
        "Phone_Usage_Purpose": [phone_usage_purpose],
        "Family_Communication": [family_comm],
        "Weekend_Usage_Hours": [weekend_usage]
    })

    st.subheader("📋 Your Input Data")
    st.write(input_data)

    # try:
    #     # If you trained with pd.get_dummies, align with training columns
    #     if training_columns is not None:
    #         input_encoded = pd.get_dummies(input_data)
    #         input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
    #         prediction = model.predict(input_encoded)[0]
    #     else:
    #         # If pipeline includes preprocessing, just pass raw input_data
    #         prediction = model.predict(input_data)[0]

    #     st.success(f"📱 Predicted Addiction Level: **{prediction:.2f}**")
    # except Exception as e:
    #     st.error(f"Prediction failed: {e}")
    try:
    # Pass raw input_data if pipeline includes preprocessing
        prediction = model.predict(input_data)[0]
        st.success(f"📱 Predicted Addiction Level: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


    # 🎨 Extra visualization
    st.bar_chart(input_data.drop(columns=["Gender","Phone_Usage_Purpose"]).T)

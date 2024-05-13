import pandas as pd 
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

def predict_status(radius_mean_BoxCox,texture_mean_BoxCox,perimeter_mean_BoxCox,area_mean_BoxCox,smoothness_mean_BoxCox,compactness_mean_BoxCox,concavity_mean_BoxCox,concavepoints_mean_BoxCox,symmetry_mean_BoxCox,fractal_dimension_mean_BoxCox):
    pickle_file_path = r"C:\python\Svmmodel.pkl"
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)

    user_data= np.array([[radius_mean_BoxCox,texture_mean_BoxCox,perimeter_mean_BoxCox,area_mean_BoxCox,smoothness_mean_BoxCox,compactness_mean_BoxCox,concavity_mean_BoxCox,concavepoints_mean_BoxCox,symmetry_mean_BoxCox,fractal_dimension_mean_BoxCox]])

    y_pred = model.predict(user_data)

    if y_pred[0] == 1:
        return 1
    else:
        return 0
    
st.set_page_config(layout= "wide")

with st.sidebar:
    option = option_menu('Select Option', options=["HOME", "PREDICT CANCER TEST"])

if option == "HOME":
    st.title("Predicting Breast Cancer")
    st.image("https://img.thedailybeast.com/image/upload/c_crop,d_placeholder_euli9k,h_1688,w_3000,x_0,y_0/dpr_2.0/c_limit,w_740/fl_lossy,q_auto/v1686072480/230606-Tran-AI-predicting-breast-cancer-tease_cziqyl")
    st.subheader("Overview:")
    st.write("This code implements a robust machine learning pipeline for breast cancer classification. It begins with thorough data preprocessing steps, including cleaning, transformation, and outlier detection, ensuring the quality of the dataset. The Exploratory Data Analysis (EDA) section provides insights into the data distribution and skewness, guiding further preprocessing decisions. Feature engineering techniques, such as feature selection, enhance model performance by selecting the most relevant features. The core of the pipeline involves building and evaluating multiple classifiers, including Support Vector Machine (SVM), Random Forest, Bagging, and AdaBoost. Through hyperparameter tuning and rigorous evaluation using various performance metrics and visualization techniques, the pipeline identifies the best-performing model. A concise summary report consolidates the model performance metrics, aiding in the selection of the most suitable classifier. Finally, the pipeline concludes with model deployment, as it saves the trained SVM model for future predictions. Overall, this pipeline offers a comprehensive and efficient approach to breast cancer classification, ensuring accuracy, reliability, and scalability in real-world applications.")
elif option == "PREDICT CANCER TEST":  # Corrected this line
    st.header("PREDICT STATUS (Malignant / Benign)")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:
        radius_mean= st.number_input(label="**Enter the Value for Radius**/ Min:6.981, Max:21.9")
        texture_mean= st.number_input(label="**Enter the Value for ITEM Texture**/ Min:9.71, Max:30.2449999999999")
        perimeter_mean= st.number_input(label="**Enter the Value for Perimeter**/ Min:43.79, Max:147.494999999999")
        area_mean= st.number_input(label="**Enter the Value for Area**/ Min:11.9791485507109, Max:39.1901109936892")
        smoothness_mean= st.number_input(label="**Enter the Value for Smoothness**/ Min:0.057975, Max:0.133695")
        
    with col2:
        compactness_mean= st.number_input(label="**Enter the Value for Compactness**/ Min:0.139212068442358, Max:0.520582452562803")
        concavity_mean= st.number_input(label="**Enter the Value for Concavity**/ Min:0.0, Max:0.645916046522211")
        concave_points_mean= st.number_input(label="**Enter the Value for Concave**/ Min:0.0, Max:0.448553229840116")
        symmetry_mean= st.number_input(label="**Enter the Value for Symmetry**/ Min:0.111199999999999, Max:0.2464")
        fractal_dimension_mean= st.number_input(label="**Enter the Value for Fractal Dimension**/ Min:0.223517337135176, Max:0.282532873501077")
        
    button= st.button("PREDICT THE STATUS", use_container_width=True)

    if button:
        status= predict_status(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean)
        
        if status == 1:
            st.write("## :green[**The Status is Malignant**]")
        else:
            st.write("## :red[**The Status is Benign**]")

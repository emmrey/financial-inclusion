import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression
import seaborn as sns
data = pd.read_csv('Financial_inclusion_dataset.csv')

#.......load the model.................
model = joblib.load(open('Financial_inclusion.pkl', 'rb'))

#....................Streamlit Development Starts.............
# st.markdown("hi style = 'text-align: right; color: FF3FA4"> EXPRESSO CHURN PREDICTION CHALLENGE<h1>, unsafe_allow_html = True)

# st.title('EXPRESSO CHURN PREDICTION CHALLENGE')
# st.write('Built by Gomycode Yellow Orange Beast')

st.markdown("<h1 style = ' color: #42f56c'>FINANCIAL INCLUSION PROJECT</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'top-margin: 0rem; color: #42d1f5'>Built By Okpara Emmanuel</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)



# st.write('Pls Enter your username')
username = st.text_input('Please enter Username:')
if st.button('Submit name'):
    st.success(f"Welcome {username}. Pls enjoy your usage")

st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #A2C579'>Project Introduction</h1>", unsafe_allow_html = True)

st.markdown("<p style = 'top margin: 0rem; 'text align: justify; color: #AED2FF'>Financial inclusion in Africa refers to the effort to provide access to a wide range of financial services to individuals and businesses across the continent, especially those who have been traditionally underserved or excluded from the formal financial system. It is a critical element of economic development and poverty reduction in Africa. Many people in Africa, particularly in rural and low-income areas, have historically been excluded from formal financial services. This exclusion is due to factors such as limited access to physical bank branches, lack of documentation, and low financial literacy. This exclusion has hindered economic growth and limited opportunities for individuals and businesses.</p>", unsafe_allow_html = True)

heat_map = plt.figure(figsize = (14,7))#........create a heatmap plot
correlation_data = data[['household_size', 'age_of_respondent', 'education_level', 'job_type', 'gender_of_respondent', 'cellphone_access', 'relationship_with_head', 'bank_account']]#.........select data for correlation
sns.heatmap(correlation_data.select_dtypes(include = 'number').corr(), annot = True, cmap = 'BuPu')

st.write(heat_map)
data.drop('uniqueid', axis = 1, inplace = True)
st.write(data.sample(5))

st.sidebar.image('image.png', width = 100, caption= f"Welcome {username}", use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)

# picture = st.camera_input('take a picture')
# if picture:
#     st.sidebar.image(picture, use_column_width= True, caption = f"welcome {username}")

st.sidebar.write('Pls decide your variable input type')
input_style = st.sidebar.selectbox('Pick Your Preferred input', ['Slider Input', 'Number Input'])

if input_style == 'Slider Input':
    household_size = st.sidebar.slider('household_size', data['household_size'].min(), data['household_size'].max())
    age_of_respondent = st.sidebar.slider('age_of_respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
    education_level = data['education_level'].unique()
    education_level = st.sidebar.select_slider('Select Your education_level', education_level)
    job_type = data['job_type'].unique()
    job_type = st.sidebar.select_slider('Select Your job_type', job_type)
    gender_of_respondent = data['gender_of_respondent'].unique()
    gender_of_respondent = st.sidebar.select_slider('Select Your gender_of_respondent', gender_of_respondent)
    cellphone_access = data['cellphone_access'].unique()
    cellphone_access = st.sidebar.select_slider('Select Your cellphone_access', cellphone_access)
    relationship_with_head = data['relationship_with_head'].unique()
    relationship_with_head = st.sidebar.select_slider('Select Your relationship_with_head', relationship_with_head)  


else:
    household_size = st.sidebar.number_input('household_size', data['household_size'].min(), data['household_size'].max())
    age_of_respondent = st.sidebar.number_input('age_of_respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
    education_level = data['education_level'].unique()
    education_level = st.sidebar.select_slider('Select Your education_level', education_level)
    job_type = data['job_type'].unique()
    job_type = st.sidebar.select_slider('Select Your job_type', job_type)
    gender_of_respondent = data['gender_of_respondent'].unique()
    gender_of_respondent = st.sidebar.select_slider('Select Your gender_of_respondent', gender_of_respondent)        
    cellphone_access = data['cellphone_access'].unique()
    cellphone_access = st.sidebar.select_slider('Select Your cellphone_access', cellphone_access)
    relationship_with_head = data['relationship_with_head'].unique()
    relationship_with_head = st.sidebar.select_slider('Select Your relationship_with_head', relationship_with_head)

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'household_size': household_size, 'age_of_respondent': age_of_respondent, 'education_level': education_level, 'job_type': job_type, 'gender_of_respondent': gender_of_respondent, 'cellphone_access': cellphone_access, 'relationship_with_head': relationship_with_head,}])

st.write(input_var)

from sklearn.preprocessing import LabelEncoder, StandardScaler
lb = LabelEncoder()
scaler = StandardScaler()

# def transformer(dataframe):
#     # scale the numerical columns
#     for i in dataframe.columns: # ---------------------------------------------- Iterate through the dataframe columns
#         if i in dataframe.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
#             dataframe[[i]] = scaler.fit_transform(dataframe[[i]]) # ------------ Scale all the numericals

#     # label encode the categorical columns
#     for i in dataframe.columns:  # --------------------------------------------- Iterate through the dataframe columns
#         if i in dataframe.select_dtypes(include = ['object', 'category']).columns: #-- Select all categorical columns
#             dataframe[i] = lb.fit_transform(dataframe[i]) # -------------------- Label encode selected categorical columns
#     return dataframe

for i in input_var.columns: # ---------------------------------------------- Iterate through the dataframe columns
    if i in input_var.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
        input_var[[i]] = scaler.fit_transform(input_var[[i]]) # ------------ Scale all the numericals

for i in input_var.columns:  # --------------------------------------------- Iterate through the dataframe columns
    if i in input_var.select_dtypes(include = ['object', 'category']).columns: #-- Select all categorical columns
        input_var[i] = lb.fit_transform(input_var[i]) # -------------------- Label encode selected categorical columns


st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        # st.write("Does this individual most likely to have or use a bank account? :", prediction)
        if prediction == 0:
            st.toast('PREDICTION SUCCESFUL !!!!')
            st.warning('This individual will not have a bank account')
    else:
        st.write('Pls press the predict button for prediction')

# with tab2:
#     st.subheader('Model Interpretation')
#     st.write(f"Churn = {model.intercept_.round(2)} + {model.coef_[0].round(2)} household_size + {model.coef_[1].round(2)} age_of_respondent + {model.coef_[2].round(2)} FREQUENCE_RECH + {model.coef_[3].round(2)} FREQUENCE + {model.coef_[4].round(2)} relationship_with_head + {model.coef_[5].round(2)} TENURE")

#     st.markdown("<br>", unsafe_allow_html= True)

#     st.markdown(f"- The expected Churn is {model.intercept_}")

#     st.markdown(f"- For every additional 1 dollar generated on household_size, the expected CHURN is expected to decrease by ${model.coef_[0].round(2)}  ")

#     st.markdown(f"- For every additional 1 dollar generated as age_of_respondent, churn is expected to increase by ${model.coef_[1].round(2)}  ")

#     st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")


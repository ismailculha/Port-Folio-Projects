import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import joblib
import base64
from IPython.core.display import HTML

st.set_page_config(
    page_title='Employee Decision Predictor',
    page_icon='icon.png'
)

st.markdown(
    """
<h2 style='text-align:center; margin-bottom: -35px; color:#d33685;'>
WILL YOUR EMPLOYEE LEAVE OR STAY ??? </h2><br>""",unsafe_allow_html=True)

st.error("As a company owner, do you want to learn whether your employees will stay with you or leave?")

st.image('employee-churn.png')   
st.info("1. Satisfaction level: Employee satisfaction point, which ranges from 0-1 😃")
st.warning("2. Evaluation Score: Evaluated performance, which also ranges from 0-1 🕵️‍♂️")
st.info("3. Number of Projects: The number of years spent by an employee in the company 2 to 7 📂")
st.warning("4. Working hours: How many hours an employee worked in a month? 100 to 300 ⏱️")
st.info("5. Years in Company: How many projects the employee is assigned to? 2 to 10 🧓")
st.warning("6. Work accident: Whether an employee has had a work accident or not 🤕")
st.info("7. Received Promotion: Has the employee had a promotion in the last 5 years 🎁")
st.warning("8. Department: Employee's working department 👩‍🔧")
st.info("9. Salary Level: Salary level of the employee; low, medium or high 💰")

st.sidebar.title("HR Analysis - Churn Prediction")


def user_input_features():
    satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.09, 1.0, 0.5)
    last_evaluation = st.sidebar.slider("Evaluation Score", 0.360, 1.0, 0.5)
    number_project = st.sidebar.selectbox('Number of Projects', [2, 3, 4, 5, 6, 7])
    average_montly_hours = st.sidebar.slider('Avg. Monthly Working Hours', min_value=100, max_value=300, value=200, step=1)
    time_spend_company = st.sidebar.selectbox('Years in the Company', [2, 3, 4, 5, 6, 7, 8, 9, 10])
    agree1 = st.sidebar.checkbox('Employee had a work accident')
    if agree1:
        Work_accident = 1
    else:
        Work_accident = 0
    agree2 = st.sidebar.checkbox('The employee received a promotion in the last 5 years')
    if agree2:
        promotion_last_5years = 1
    else:
        promotion_last_5years = 0
    choice = st.sidebar.selectbox('Department',
        ("Information Technology (IT)",
        "Research and Development (R & D)",
        "Accounting",
        "Human Resources",
        "Management",
        "Marketing",
        "Product Management",
        "Sales",
        "Support",
        "Technical"),
    )
    if choice == "Information Technology (IT)":
        departments = "IT"
    elif choice == "Research and Development (R & D)":
        departments = "RandD"
    elif choice == "Accounting":
        departments = "accounting"
    elif choice == "Human Resources":
        departments = "hr"
    elif choice == "Management":
        departments = "management"
    elif choice == "Marketing":
        departments = "marketing"
    elif choice == "Product Management":
        departments = "product_mng"
    elif choice == "Sales":
        departments = "sales"
    elif choice == "Support":
        departments = "support"
    elif choice == "Technical":
        departments = "technical" 
    choice2 = st.sidebar.radio('Salary Level', ["Low", "Medium", "High"])
    if choice2 == "Low":
        salary = "low"
    elif choice2 == "Medium":
        salary = "medium"
    elif choice2 == "High":
        salary = "high"
    new_df = {"satisfaction_level":satisfaction_level,
              "last_evaluation":last_evaluation,
              "number_project":number_project,
              "average_montly_hours":average_montly_hours,
              "time_spend_company":time_spend_company,
              "Work_accident":Work_accident,
              "promotion_last_5years":promotion_last_5years,
              "departments":departments,
              "salary":salary}
    features = pd.DataFrame(new_df, index=[0])
    return features
input_df = user_input_features()

st.markdown("""<h4 style='text-align:left; color:#d33685;'>Employee Features</h4>
""", unsafe_allow_html=True
)
resa = input_df.rename(columns={"satisfaction_level":"Satisfaction Level",
              "last_evaluation":"Evaluation Score",
              "number_project":"Number of Projects",
              "average_montly_hours":"Working hours",
              "time_spend_company":"Years in Company",
              "Work_accident":"Work Accident",
              "promotion_last_5years":"Received Promotion",
              "departments":"Department",
              "salary":"Salary Level"})
resa['Working hours'] = resa['Working hours'].astype('int')
resa['Work Accident'] = resa['Work Accident'].map({0:'No', 1:'Yes'})
resa['Received Promotion'] = resa['Received Promotion'].map({0:'No', 1:'Yes'})
resa['Department'] = resa['Department'].map({'IT':'Information Technology',
                        'RandD': 'R & D',
                        'accounting':'Accounting',
                        'hr':'Human Resources',
                        'management':'Management',
                        'marketing':'Marketing',
                        'product_mng':'Product Management',
                        'sales':'Sales',
                        'support':'Support',
                        'technical':'Technical'})
resa['Salary Level'] = resa['Salary Level'].map({'low':'Low', 'medium':'Medium', 'high':'High'})

st.write(HTML(resa.to_html(index=False, justify='left')))

                             
g4_model= joblib.load(open("xgb_model_final.pkl","rb"))

st.subheader("Press predict if the information is okay")
prediction = g4_model.predict(input_df)


if st.button("Predict"): 
    if prediction==0:
        st.success(prediction[0])
        st.success(f"Employee will STAY")
    elif prediction==1:
        st.warning(prediction[0])
        st.warning(f"Employee will LEAVE")



m = st.markdown("""<style>
div.stButton > button:first-child {
    background-color: #ea9999;
    color: white;
    height: 5em;
    width: 15em;
    border-radius:20px;
    border:3px solid #000000;
    font-size:20px;
    font-weight: bold;
    margin: auto;
    display: block;
}

# div.stButton > button:hover {
# 	background:linear-gradient(to bottom, #ef7676 5%, #ef7676 100%);
# 	background-color:#ef7676;
# }
# div.stButton > button:active {
# 	position:relative;}
#  <style>""", unsafe_allow_html=True)




                               

     

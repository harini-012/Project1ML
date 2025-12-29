import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
st.title("Medical Insurance cost prediction")
st.caption("Get an estimated annual insurance cost based on your details")
data=pd.read_csv("insurance.csv")
data['sex']=data['sex'].map({'male':0,'female':1})
data['smoker']=data['smoker'].map({'no':0,'yes':1})
X=data[['age','sex','bmi','children','smoker']]
y=data['charges']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
basic_plan=LinearRegression()
basic_plan.fit(X_train,y_train)
premium_plan=Ridge(alpha=1.0)
premium_plan.fit(X_train,y_train)
st.header("Enter the details of the patient")
age = st.text_input("Age")
gender = st.selectbox("Gender", ["Select", "Male", "Female"])
bmi = st.text_input("BMI")
children = st.text_input("Number of Dependents")
smoker = st.selectbox("Smoking Status", ["Select", "Non-Smoker", "Smoker"])
if st.button("Get Cost Estimate"):
     if not age or not bmi or not children or gender == "Select" or smoker == "Select":
        st.warning("⚠️ Please fill in all the details before proceeding.")
     else:
        age = int(age)
        bmi = float(bmi)
        children = int(children)
        sex = 0 if gender == "Male" else 1
        smoker_val = 0 if smoker == "Non-Smoker" else 1
        user=[[age,sex,bmi,children,smoker_val]]
        estimate_basic=basic_plan.predict(user)[0]
        estimate_premium=premium_plan.predict(user)[0]
        st.markdown("📊Estimated Annual insurance cost")
        col1,col2=st.columns(2)
        with col1:
            st.metric("Standard Plan Estimate",f"₹{estimate_basic:,.0f}")
        with col2:
            st.metric("Enhanced plan Estimate",f"₹{estimate_premium:,.0f}")
        st.info("💡 This is an estimated cost based on historical insurance data.")
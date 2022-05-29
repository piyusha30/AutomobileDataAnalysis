# -*- cod"ing: utf"-8 -*-
""""
Created on Tue May 24 19:15:29 2022

@author: Piyusha Nair
"""
import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

#main page
#sidebar
st.set_page_config(
        page_title="Automobile Data Analysis",
)
st.sidebar.title("EXPLORE / PREDICT")
page=st.sidebar.selectbox("Explore OR Predict",("Predict","Explore"))

if page=="Predict":
    show_predict_page()#calling predict page
else :
    show_explore_page()#calling explore page

# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:00:52 2022

@author: Piyusha Nair
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from predict_page import load_data 
from predict_page import clean_data
from predict_page import clean_car_sales_data_
from predict_page import load_carsales

#pie chart for Cars Launched By Company 
def pie_chart(cleaned_data) :
    #pi chart of available cars
    colors =  sns.color_palette('bright')[0:5]#.color palette
    temp = cleaned_data['Make'].value_counts()
    temp2 = temp.head(20)#only collecting top 20 cars to display
    
    if len(temp) > 10:
       temp2['remaining {0} items'.format(len(temp)-10)] = sum(temp[10:])
    
    temp2.plot(kind='pie',autopct="%1.1f%%",shadow=False,fontsize=15,pctdistance=0.9,colors=colors,wedgeprops={"edgecolor":"0","linewidth":0.5,"linestyle":"solid","antialiased":True},figsize=(10,10)) 
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Cars Launched By Company')
    #plotting on streamlit
    st.pyplot()
    st.write("The above graph provides information about data of cars available by researcher.")

# Graph for showing how features of cars are correlated.
def heat_map(cleaned_data) :
     #A heatmap (aka heat map) depicts values for a main variable of interest across two axis variables as a grid of colored squares
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cleaned_data.corr(), center=0, cmap='BrBG', annot=True)
    st.subheader('Feature Correlation')
    st.pyplot()
    st.write("The above graph gives an idea about how features of cars are correlated.")

#Graph of various body types and it's count available
def body_type_estimate(cleaned_data) :
    #histogram of body type and count\
    plt.figure(figsize=(16,8))
    sns.countplot(data=cleaned_data,x='Body_Type',color='red')
    plt.xticks(fontsize=8)
    plt.xlabel('Body type')
    plt.ylabel('Count')
    plt.title("Body type ")
    plt.show()
    st.subheader('Body Type Analysis')
    st.pyplot()  
    st.write("The above graph provides information about various body types and it's count available.")
#Graph of various fuel types of cars and their counts.
def fuel_type_estimate(cleaned_data) :
    #sub plot of fuel type
    pd.crosstab(cleaned_data.Make,cleaned_data['Fuel_Type'],margins=True).T.style.background_gradient(cmap='summer_r')
    f,ax=plt.subplots(1,2,figsize=(16,7))
    cleaned_data['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title('Fuel Type')
    ax[0].set_ylabel('Count')
    sns.countplot('Fuel_Type',data=cleaned_data,ax=ax[1],order=cleaned_data['Fuel_Type'].value_counts().index)
    ax[1].set_title('Fuel Type')
    st.subheader('Fuel Type Estimate')
    st.pyplot() 
    st.write("The above graph provides information about various fuel types of cars and their counts.")

#Graph about manufacturer and count of cars maufactured by them.
def manufacturer_estimate(cleaned_data) :
    plt.subplots(figsize=(16,8))
    ax=cleaned_data['Make'].value_counts().plot.bar(width=0.9)
    for p in ax.patches:
        ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
    plt.ylabel('Automobile Manufacturer',fontsize=20)
    plt.xlabel('Number of Cars',fontsize=20)
    plt.title('Count Of Cars Manufactured By Manufacturer',fontsize=30)
    ax.tick_params(labelsize=15)
    st.subheader('Maufacturer Estimate')
    plt.show()
    st.pyplot() 
    st.write("The above graph provides information about manufacturer and count of cars maufactured by them.")
# Graph of emission norms of cars it's count   
def emission_estimate(cleaned_data)  :
    #subplot of emission type
#sub plot of fuel type
    pd.crosstab(cleaned_data.Make,cleaned_data['Emission_Norm'],margins=True).T.style.background_gradient(cmap='summer_r')
    
    f,ax=plt.subplots(1,2,figsize=(16,7))
    cleaned_data['Emission_Norm'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title('Emission_Norm')
    ax[0].set_ylabel('Count')
    sns.countplot('Emission_Norm',data=cleaned_data,ax=ax[1],order=cleaned_data['Emission_Norm'].value_counts().index)
    ax[1].set_title('Emission_Norm')
    st.subheader('Emisssion norm Analysis')
    st.pyplot()
    st.write("The above graph provides information about emission norms of cars it's count.")
#Graph about launch month of various graphs.
def latest_launch_month(new_sales_data) :
    sns.catplot(x = 'Latest_Launch_month', data = new_sales_data, kind= 'count')
    plt.xlabel('Launch Month')
    plt.ylabel('Count Of Cars')
    st.pyplot()
    st.write("The above graph provides information about launch month of various graphs.It can be noted that maximum cars were launched in October(10)")

#Graph about number of sales of cars in different months.
def month_vs_sales(new_sales_data):
    plt.figure(figsize=(16,8))
    sns.barplot(x='Latest_Launch_month', y='Sales_in_thousands', data = new_sales_data)
    st.pyplot()
    st.write("The above graph provides information about number of sales of cars in different months.")

#Graph about how features of cars and customers are correlated.
def cust_heat_map(new_sales_data) :
    plt.figure(figsize=(16,8))
    sns.heatmap(new_sales_data.corr(), cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap = 'Blues')
    st.pyplot() 
    st.write("The above graph gives an idea about how features of cars and customers are correlated.")

#main function which will be called by app page
def show_explore_page() :
    st.title("AUTOMOBILE  DATA  EXPLORATION")
    st.markdown("""---""")
    automobile_data=load_data()
    cleaned_data=clean_data(automobile_data)
    
    option= st.selectbox(
        'Explore various plots!',
        ('Cars Launched By Company','Count Of Cars By Manufacturer','Feature Correlation','Emisssion Norm Stats','Fuel Type Analysis','Car Body Type Analysis'))
    st.write('You have selected',option)
    if (option=='Cars Launched By Company'):
        pie_chart(cleaned_data)
    if (option=='Feature Correlation'):
        heat_map(cleaned_data)
    if (option=='Emisssion Norm Stats'):
        emission_estimate(cleaned_data)
    if (option=='Fuel Type Analysis'):
        fuel_type_estimate(cleaned_data)
    if (option=='Count Of Cars By Manufacturer'):
        manufacturer_estimate(cleaned_data)
    if(option=='Car Body Type Analysis') :
        body_type_estimate(cleaned_data)
    st.markdown("""---""")
    
    st.subheader('Customer Car Sales Data Exploaration')
    car_sale=load_carsales()
    new_sales_data=clean_car_sales_data_(automobile_data,car_sale)
    
    option= st.selectbox(
        'Explore Various Plots !',
        ('Count Of Cars By Launch Month','Sales By Launch Month'))
    
    if (option=='Count Of Cars By Launch Month'):
        latest_launch_month(new_sales_data) 
    if (option=='Sales By Launch Month'):
        month_vs_sales(new_sales_data)
    st.markdown("""---""")
   
   
        
        
        


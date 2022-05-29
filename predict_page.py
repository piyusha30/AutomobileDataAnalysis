# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:15:19 2022

@author: Piyusha Nair
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import seaborn as sns
#from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')

#loading Engage datasetr
def load_data():
    automobile_data=pd.read_csv("cars_engage_2022 (1).csv",low_memory=False)
    return automobile_data
#load month predictor pickle
def loadModel_monthPred():
    with open('saved_monthpred.pkl','rb') as file :
        data=pickle.load(file)
    return data
#load  car clustering pickle
def loadModel_carPred():
   # with open
   with open('car_pred.pkl','rb') as file :
       data_pickle=pickle.load(file)
   return data_pickle
#loading random forest pickle file
def load_carSales_pickle() :
    with open('actual_month_pred.pkl','rb') as file :
     data=pickle.load(file)
    return data
#loading linear regression pickle filee
def load_Sales_predict_pickle() :
    with open('sales_pred.pkl','rb') as file :
     data=pickle.load(file)
    return data
def load_carsales() :
   car_sales=pd.read_csv("Car_sales (1).csv",low_memory=False)
   return car_sales
     
    

#cleaning automobile dataset
def clean_data(automobile_data) :
    #cleaning numeric data
    def truncateAttribute(symbol,attribute) :
        automobile_data[attribute] =automobile_data[attribute].replace({symbol: str()}, regex=True).astype(float)
    def truncateMultipleAttribute(symbol1,symbol2,attribute) :
         automobile_data[attribute] =automobile_data[attribute].replace({symbol1: str(), symbol2 :str()}, regex=True).astype(float)
    truncateMultipleAttribute('Rs. ',',','Ex-Showroom_Price')
    truncateAttribute(' cc','Displacement')
    truncateAttribute(' mm','Height')
    truncateAttribute(' mm','Length')
    truncateAttribute(' mm','Width')
    truncateAttribute(' mm','Wheelbase')
    #splliting data into numeric and categorical
    numeric_data_1=automobile_data.select_dtypes(include=[np.float])
    numeric_data_1.fillna(numeric_data_1.median(), inplace=True)
    numeric_data2 = automobile_data.select_dtypes(include=[np.int64])
    numeric_data = pd.concat([numeric_data_1, numeric_data2], axis=1)
    #replacing misssing values
    categorical_data = automobile_data.select_dtypes(exclude=[np.int64])
    categorical_data1 = categorical_data.select_dtypes(exclude=[np.float64])
    categorical_data2 = categorical_data1.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
    full_data = pd.concat([categorical_data2, numeric_data], axis=1)
    
    return full_data
#cleaning car sales dataset
def clean_car_sales_data_(automobile_data,car_sale) :
    car_sale.fillna(car_sale.median(), inplace=True)
    car_sale[["Latest_Launch_month", "Latest_Launch_day", "Latest_Launch_year"]] = car_sale["Latest_Launch"].str.split("/", expand = True)
    auto_mobile=load_data()
    automobile_data=clean_data(auto_mobile)
    
    automobile_data['Minimum_Turning_Radius'] = automobile_data['Minimum_Turning_Radius'].astype(str).str.replace('meter', '')
    automobile_data['Gross_Vehicle_Weight'] = automobile_data['Gross_Vehicle_Weight'].astype(str).str.replace('kg', '')
    automobile_data['Fuel_Tank_Capacity'] = automobile_data['Fuel_Tank_Capacity'].astype(str).str.replace('litres', '')
    automobile_data['City_Mileage'] = automobile_data['City_Mileage'].astype(str).str.replace('km/litre', '')
    automobile_data['Highway_Mileage'] = automobile_data['Highway_Mileage'].astype(str).str.replace('km/litre', '')
    automobile_data['ARAI_Certified_Mileage'] = automobile_data['ARAI_Certified_Mileage'].astype(str).str.replace('km/litre', '')
    automobile_data['Kerb_Weight'] = automobile_data['Kerb_Weight'].astype(str).str.replace('kg', '')
    automobile_data['Ground_Clearance'] = automobile_data['Ground_Clearance'].astype(str).str.replace('mm', '')
    automobile_data['Front_Track'] = automobile_data['Front_Track'].astype(str).str.replace('mm', '')
    automobile_data['Rear_Track'] = automobile_data['Rear_Track'].astype(str).str.replace('mm', '')
    automobile_data['Boot_Space'] = automobile_data['Boot_Space'].astype(str).str.replace('litres', '')
    
    automobile_data['Minimum_Turning_Radius'] = pd.to_numeric(automobile_data['Minimum_Turning_Radius'],errors = 'coerce')
    automobile_data['Gross_Vehicle_Weight'] = pd.to_numeric(automobile_data['Gross_Vehicle_Weight'],errors = 'coerce')
    automobile_data['Fuel_Tank_Capacity'] = pd.to_numeric(automobile_data['Fuel_Tank_Capacity'],errors = 'coerce')
    automobile_data['City_Mileage'] = pd.to_numeric(automobile_data['City_Mileage'],errors = 'coerce')
    automobile_data['Highway_Mileage'] = pd.to_numeric(automobile_data['Highway_Mileage'],errors = 'coerce')
    automobile_data['ARAI_Certified_Mileage'] = pd.to_numeric(automobile_data['ARAI_Certified_Mileage'],errors = 'coerce')
    automobile_data['Kerb_Weight'] = pd.to_numeric(automobile_data['Kerb_Weight'],errors = 'coerce')
    automobile_data['Ground_Clearance'] = pd.to_numeric(automobile_data['Ground_Clearance'],errors = 'coerce')
    automobile_data['Front_Track'] = pd.to_numeric(automobile_data['Front_Track'],errors = 'coerce')
    automobile_data['Rear_Track'] = pd.to_numeric(automobile_data['Rear_Track'],errors = 'coerce')
    automobile_data['Boot_Space'] = pd.to_numeric(automobile_data['Boot_Space'],errors = 'coerce')     
    
    numeric_data_1=automobile_data.select_dtypes(include=[np.float])
    numeric_data_1.fillna(numeric_data_1.median(), inplace=True)
    numeric_data2 = automobile_data.select_dtypes(include=[np.int64])
    numeric_data = pd.concat([numeric_data_1, numeric_data2], axis=1)
    
    categorical_data = automobile_data.select_dtypes(exclude=[np.int64])
    categorical_data1 = categorical_data.select_dtypes(exclude=[np.float64])
    #features to get dummies
    cat_feats = ['Cruise_Control', 'ASR_/_Traction_Control', 'Automatic_Headlamps', 'Leather_Wrapped_Steering', 
             'Paddle_Shifters', 'Rain_Sensing_Wipers', 'ISOFIX_(Child-Seat_Mount)',
             'Turbocharger', 'Cooled_Glove_Box', 'ESP_(Electronic_Stability_Program)', 'iPod_Compatibility', 'Tyre_Pressure_Monitoring_System', 
            'Second_Row_AC_Vents', 'Navigation_System', 'EBA_(Electronic_Brake_Assist)', 'Average_Speed', 'Cigarette_Lighter',
            'USB_Compatibility', 'Key_Off_Reminder', 'Gear_Shift_Reminder', 'Fasten_Seat_Belt_Warning', 'EBD_(Electronic_Brake-force_Distribution)',
            'Door_Ajar_Warning', 'ABS_(Anti-lock_Braking_System)', 'Seat_Back_Pockets', 'Engine_Immobilizer', 'Gear_Indicator',
            'Hill_Assist', 'Auto-Dimming_Rear-View_Mirror', 'Multifunction_Display', 'Low_Fuel_Warning', 'FM_Radio', 'Engine_Malfunction_Light',
            'Distance_to_Empty', 'Child_Safety_Locks', 'Central_Locking', 'CD_/_MP3_/_DVD_Player', 'Bluetooth', 'Average_Fuel_Consumption',
            'Aux-in_Compatibility', 'Start_/_Stop_Button']
    categorical_data1 = pd.get_dummies(categorical_data1, columns = cat_feats, 
                                    dummy_na = False)
    categorical_data2 = categorical_data1.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
    #non useful data,hence eliminated
    categorical_data2.drop(["Front_Tyre_&_Rim"], axis=1, inplace=True)
    categorical_data2.drop(["Rear_Tyre_&_Rim"], axis=1, inplace=True)
    categorical_data2.drop(["Front_Suspension"], axis=1, inplace=True)
    categorical_data2.drop(["Rear_Suspension"], axis=1, inplace=True)
    categorical_data2.drop(["Airbags"], axis=1, inplace=True)
    categorical_data2.drop(["Seat_Height_Adjustment"], axis=1, inplace=True)
    categorical_data2.drop(["Basic_Warranty"], axis=1, inplace=True)
    categorical_data2.drop(["Extended_Warranty"], axis=1, inplace=True)
    categorical_data2.drop(["Wheels_Size"], axis=1, inplace=True)
    categorical_data2.drop(["Tripmeter"], axis=1, inplace=True)
    categorical_data2.drop(["12v_Power_Outlet"], axis=1, inplace=True)
    categorical_data2.drop(["Variant"], axis=1, inplace=True)
    categorical_data2.drop(["Power"], axis=1, inplace=True)
    full_data = pd.concat([categorical_data2, numeric_data], axis=1)
    
    #left merging automobile and sales data
    merged_left = pd.merge(left=categorical_data2, right=car_sale, how='left', left_on='Model', right_on='Model')
    merged_left['Sales_in_thousands'].fillna(merged_left['Sales_in_thousands'].median(), inplace=True)
    merged_left['Price_in_thousands'].fillna(merged_left['Price_in_thousands'].median(), inplace=True)
    merged_left['Latest_Launch_month'].fillna(merged_left['Latest_Launch_month'].mode()[0], inplace=True)
    merged_left['Latest_Launch_year'].fillna(merged_left['Latest_Launch_year'].mode()[0], inplace=True)
    merged_left['Sales_in_thousands'].fillna(merged_left['Sales_in_thousands'].median(), inplace=True)
    merged_left['Latest_Launch_month'].fillna(merged_left['Latest_Launch_month'].mode()[0], inplace=True)
    new_data=pd.concat([merged_left,full_data],axis=1)
    
    return new_data
   
#mapping categorical data
def map_data(full_data):
    ord_map_fuel_type={'Diesel':1,'Petrol':2,'CNG':3,'Hybrid':4,'Electric':5,'CNG + Petrol':6}
    full_data['fuel_type_mapped']=full_data['Fuel_Type'].map(ord_map_fuel_type)
   
    
    ord_map_body_type={'Hatchback':1, 'MPV':2, 'MUV':3, 'SUV':4, 'Sedan':5, 'Crossover':6, 'Coupe':7, 'Convertible':8
     ,'Sports, Hatchback':9, 'Sedan, Coupe':10, 'Sports':11, 'Crossover, SUV':12,
     'SUV, Crossover':13, 'Sedan, Crossover':14, 'Sports, Convertible':15, 'Pick-up':16,
     'Coupe, Convertible':17}
    full_data['body_type_mapped']=full_data['Body_Type'].map(ord_map_body_type)
    
    #emission norm: Emission_Norm, 
    ord_map_emission_norm={'BS IV':1, 'BS 6':2, 'BS III':3, 'BS VI':4}
    full_data['emission_norm_mapped']=full_data['Emission_Norm'].map(ord_map_emission_norm)
   
    #type : ['Manual' 'Automatic' 'AMT' 'CVT' 'DCT']
    ord_map_type={'Manual':1, 'Automatic':2,'AMT':3, 'CVT':4 ,'DCT':5}
    full_data['type_mapped']=full_data['Type'].map(ord_map_type)
   
    
    #CLEAN GEARS ['4' '5' '6' '7' '9' '8' 'Single Speed Reduction Gear']
    ord_map_gears={'4':4, '5':5, '6':6, '7':7 ,'9':9, '8':8 ,'Single Speed Reduction Gear':1}
    full_data['gears_mapped']=full_data['Gears'].map(ord_map_gears)
    return full_data
#function will cleaning, mapping and selects important features to feed to the model
def clean_concat(automobile_data):
    full_data=clean_data(automobile_data)
    full_data=map_data(full_data)
    data=full_data[["Ex-Showroom_Price","Displacement","Cylinders","fuel_type_mapped","Height","Length"
               ,"Width", "body_type_mapped","type_mapped",'emission_norm_mapped',"gears_mapped","Seating_Capacity","Number_of_Airbags"]]
    return data
#finding number of clusters of car features by elbow method
def define_clusters(data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    st.pyplot()
    st.write("Here since the elbow of the above graph arrives at 3, we will consider 3 groups!")
#demapping the features to display proper format to the user
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return 3
def demap_fuel_type_mapped(demapped_df) :
    ord_demap_fuel_type={1:'Diesel',2:'Petrol',3:'CNG',4:'Hybrid',5:'Electric',6:'CNG + Petrol'}
   
    demapped_df['fuel_type_mapped']= demapped_df['fuel_type_mapped'].map(ord_demap_fuel_type)
#demapping body type   
def Demap_body_type_mapped(demapped_df):
    ord_map_body_type={1:'Hatchback',2:'MPV',3: 'MUV',4:'SUV',5:'Sedan',6:'Crossover',7:'Coupe',8:'Convertible'
 ,9:'Sports, Hatchback',10: 'Sedan, Coupe',11: 'Sports', 12:'Crossover, SUV',
 13:'SUV, Crossover',14: 'Sedan, Crossover',15: 'Sports, Convertible',16: 'Pick-up',
 17:'Coupe, Convertible'}
    
    demapped_df['body_type_mapped']= demapped_df['body_type_mapped'].map(ord_map_body_type)
#demapping emission norm     
def Demap_emission_norm_mapped(demapped_df):
    ord_map_emission_norm={1:'BS IV',2:'BS 6',3:'BS III',4:'BS VI'}
    
    demapped_df['emission_norm_mapped']= demapped_df['emission_norm_mapped'].map(ord_map_emission_norm)
#demapping type  
def Demap_type_mapped(demapped_df):
    ord_demap_type={1:'Manual',2: 'Automatic',3:'AMT',4: 'CVT',5:'DCT'}
    
    demapped_df['type_mapped']= demapped_df['type_mapped'].map(ord_demap_type)
#demapping gears
def Demap_gears(demapped_df) :
    ord_map_gears={4:'4',5:'5',6:'6',7:'7',9:'9',8:'8',1:'Single Speed Reduction Gear'}
    
    demapped_df['gears_mapped']= demapped_df['gears_mapped'].map(ord_map_gears)
#changing type to int for some features   
def round_off(demapped_df) :
 
  demapped_df["Cylinders"]=demapped_df["Cylinders"].astype(int)
  demapped_df['gears_mapped']=demapped_df['gears_mapped'].astype(int)
  demapped_df['type_mapped']=demapped_df['type_mapped'].astype(int)
  demapped_df['emission_norm_mapped']=demapped_df['emission_norm_mapped'].astype(int)
  demapped_df['body_type_mapped']=demapped_df['body_type_mapped'].astype(int) 
  demapped_df['fuel_type_mapped']=demapped_df['fuel_type_mapped'].astype(int)
  demapped_df['Number_of_Airbags']=demapped_df['Number_of_Airbags'].astype(int)
  demapped_df['Seating_Capacity']=demapped_df['Seating_Capacity'].astype(int)
#function for displaying month name
def month_determiner(predict) :
    if predict=='1' :
        return 'January'
    if predict=='2' :
        return 'Feburary'
    if predict=='3' :
        return 'March'
    if predict=='4' :
        return 'April'
    if predict=='5' :
        return 'May'
    if predict=='6' :
        return 'June'
    if predict=='7' :
        return 'July'
    if predict=='8' :
        return 'August'
    if predict=='9' :
        return 'September'
    if predict=='10' :
        return 'October'
    if predict=='11' :
        return 'November'
    else :
      return 'December'
 #selecting top 5 features through random forest classifier
def select_top_5_features(new_data):
    # random forest for feature importance on a classification problem
    # define dataset
    
    X = new_data.drop('Latest_Launch_month', axis=1)
    y = new_data['Latest_Launch_month']
    # define dataset
    X, y = make_classification(n_samples=1276, n_features=250, n_informative=5, n_redundant=5, random_state=1)
# define the model
    model = RandomForestClassifier()
# fit the model
    model.fit(X, y)
    plt.figure(figsize=(16,8))
    feat_importances = pd.Series(model.feature_importances_, new_data.columns)
    red = feat_importances.nlargest(5).plot(kind='barh', figsize=(20,20))
    plt.title("Top 5 important features",fontsize=25)
    plt.xlabel("Count")
    plt.yticks(fontsize=20)
    st.pyplot()
    st.write("Here , we can see that these 5 features contribute the most.")
    #selecting top 50 attributes
    filtered_data = new_data[[ 'Navigation_System_Yes', 'Emission_Norm','Start_/_Stop_Button_Yes','Turbocharger_Yes','Engine_Location']]
    filtered_data = filtered_data.replace({
                "Emission_Norm":  {"BS IV": 1, "BS 6": 2, "BS VI": 3, "BS III": 4},
                
                "Engine_Location": {"Front, Transverse": 1, "Front, Longitudinal": 2, "Rear, Transverse": 3, "Rear Mid, Transverse": 4, "Mid, Longitudinal": 5, "Mid, Transverse": 6, "Rear, Longitudinal": 7},
                })
    filtered_data['Sales_in_thousands']=new_data[['Sales_in_thousands']]
    filtered_data['Latest_Launch_month'] = new_data['Latest_Launch_month'].astype('category')
    return filtered_data

#main funtion which will be called by app page
def show_predict_page()  :
    st.title("AUTOMOBILE  DATA  ANALYSIS")
    st.markdown("""---""")
    #loading and cleaning data
    automobile_data=load_data()
    full_data=clean_data(automobile_data)
    cleaned_data=clean_concat(automobile_data)
    #calling clustering model(Kmeans)
    kmeans_model=loadModel_carPred()
    data=loadModel_monthPred()
    kmeans1=kmeans_model['model'] 
    month_predictor=data['model']
    label=kmeans1.fit_predict(cleaned_data)
    st.subheader("Car Segmentation Based on Popular Features :") 
    st.subheader("No. Of Clusters Determined By Elbow Method")
    #calling elbow method
    define_clusters(cleaned_data)
    #plotting similar clustering
    plt.figure(figsize=(16,10))
    clustered_0=cleaned_data[label == 0]
    clustered_1=cleaned_data[label==1]
    clustered_2=cleaned_data[label==2]


    sns.scatterplot(data=cleaned_data, x=clustered_0['Ex-Showroom_Price'], y=clustered_0['Displacement'], style=full_data['Fuel_Type'],s=80,color="red",label="Group 0")
    sns.scatterplot(data=cleaned_data, x=clustered_1['Ex-Showroom_Price'], y=clustered_1['Displacement'], style=full_data['Fuel_Type'],s=80,color='green',label="Group 1")
    sns.scatterplot(data=cleaned_data, x=clustered_2['Ex-Showroom_Price'], y=clustered_2['Displacement'], style=full_data['Fuel_Type'],s=80,color='blue',label="Group 2")
    plt.xlabel('Showroom Price')
    plt.tick_params(axis='x', labelsize=8)
    plt.ylabel('Displacement')
    #plotting cluster centroids
    centroids = kmeans1.cluster_centers_
     
    #plotting the results:
    
    plt.scatter(centroids[:,0] , centroids[:,1] ,marker='^', s = 380, color = 'black')
    
    #to avoid repetition of lavles
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    st.subheader("Segmentation By Similar Attributes : ")
    st.pyplot()
    #printing data frame
    label=["Ex-Showroom_Price","Displacement","Cylinders","fuel_type_mapped","Height","Length"
               ,"Width", "body_type_mapped","type_mapped",'emission_norm_mapped',"gears_mapped","Seating_Capacity","Number_of_Airbags"]
    st.subheader("  Popular Car  Features : ")
    temp_data= pd.DataFrame(kmeans1.cluster_centers_, columns = label)
    temp_df=pd.DataFrame(temp_data)
    # Create the pandas DataFrame
    temp_df = pd.DataFrame(kmeans1.cluster_centers_, columns = label)
    #creating a copy of df
    demapped_df=temp_df.copy()
    round_off(demapped_df)
    #dammping attributes
    Demap_type_mapped(demapped_df)
    demap_fuel_type_mapped(demapped_df)
    Demap_gears(demapped_df)
    Demap_body_type_mapped(demapped_df)
    Demap_emission_norm_mapped(demapped_df)
    #displaying dataframe in proper format
    st.dataframe(demapped_df.style.highlight_max(axis=0))
    print(demapped_df)
    #inserting temp_df["month"]=0 before feediong to the model
    temp_df.insert(13, 'month', 0)
   
    #FOR PREDICTING MONTH for each group
    st.subheader("Predict Best Launch Month For Each Group")
    #group 0
    ok_1 =st.button("Launch Month For Group 0")
    if ok_1:
        
        input_data=temp_df.iloc[0]
        
        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        prediction = month_predictor.predict(input_data_reshaped)
        print('Month:')
        print(prediction[0])
    
        st.write("The Estimated Launch Month For Group 0 is :")
        st.subheader(month_determiner(prediction[0]))
    #group 1
    ok_2 =st.button("Launch Month For Group 1")
    if ok_2:
        input_data = temp_df.iloc[1]

        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        prediction = month_predictor.predict(input_data_reshaped)
        print('Month:')
        print(prediction[0])
    
        st.write("The Estimated Launch Month For Group 1 is :")
        st.subheader(month_determiner(prediction[0]))
    #group 2
    ok_3 =st.button("Launch Month For Group 2")
    if ok_3:
        input_data =temp_df.iloc[2]

        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        prediction = month_predictor.predict(input_data_reshaped)
        print('Month:')
        print(prediction[0])
    
        st.write("The Estimated Launch Month For Group 2 is")
        st.subheader(month_determiner(prediction[0]))
    st.markdown("""---""")
    #calling linear regression model for predicting sales
    st.subheader('Do you want to predict sales?:')
    sales_predictor=load_Sales_predict_pickle()
    sales_predictor_model=sales_predictor['model']
    
    int_val_year = st.slider('Select Year:', min_value=1990, max_value=2050, value=1990, step=1)
    X=np.array([[int_val_year]])
    #calling regressor
    y_pred = sales_predictor_model.predict(X).astype(int)
    sales_button=st.button("Calculate sales")
    st.subheader('Sales for the year : ')
    st.subheader(y_pred[0][0])       
    st.markdown("""---""")
    #to predict launch month for customisable features : 
    st.subheader('Predict Launch Month For Your Dream Car!')
    car_sale=load_carsales()
    new_sales_data=clean_car_sales_data_(automobile_data,car_sale)  

    st.subheader("Top 5 features :")#features which contributes the most
    imp_sales_data=select_top_5_features(new_sales_data)
    
    
    
    st.write('Choose Specifications: ')
    emission=st.selectbox('Emission System',("BS IV","BS 6","BS VI","BS III"))
    if(emission=="BS IV") :
        emission_norm=1
    if(emission=="BS 6") :
        emission_norm=2
    if(emission=="BS VI") :
        emission_norm=3
    if(emission=="BS III") :
        emission_norm=4
    
    navigation=st.selectbox('Navigation System',("Yes","No"))
    if(navigation=="Yes") :
        navigation_yes=1
    if(navigation=="No") :
        navigation_yes=0
        
    start_stop=st.selectbox('Start Stop button',("Yes","No"))
    if(start_stop=="Yes") :
        start_stop_yes=1
    if(start_stop=="No") :
        start_stop_yes=0
    turbo_charger=st.selectbox('Turbo Charger button',("Yes","No"))
    if(turbo_charger=="Yes") :
        turbo_charger_yes=1
    if(turbo_charger=="No") :
        turbo_charger_yes=0
    engine=st.selectbox('Emission System',("Front, Transverse","Front, Longitudinal","Rear, Transverse","Rear Mid, Transverse","Mid, Longitudinal","Mid, Transverse","Rear, Longitudinal"))
    if(engine=="Front, Transverse") :
        engine_loc=1
    if(engine=="Front, Longitudinal") :
        emission_norm=2
    if(engine=="Rear, Transverse") :
        engine_loc=3
    if(engine=="Rear Mid, Transverse") :
        engine_loc=4
    if(engine=="Mid, Longitudinal") :
        engine_loc=5
    if(engine=="Mid, Transverse") :
        engine_loc=6
    if(engine=="Rear, Longitudinal") :
        engine_loc=7
    
    price=st.number_input('Enter expected sale ')
    #calling  random forest tree model from pickle
    month_predict=load_carSales_pickle()
    month_predictor=month_predict['model']
    input_data = (navigation_yes,emission_norm,start_stop_yes,turbo_charger_yes,engine_loc,price)   
    input_data_as_numpy_array = np.asarray(input_data)
    #reshaping data before feeding to model
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    calc_month =st.button("Calculate month for ")
    if calc_month:
        prediction = month_determiner(month_predictor.predict(input_data_reshaped))
        st.subheader("The estimated month for above specifications is : ")
        st.subheader(month_determiner(prediction))
    
    st.markdown("""---""")
    

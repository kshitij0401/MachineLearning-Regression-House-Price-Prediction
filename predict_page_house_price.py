import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('/Users/kshitijsaxena/Desktop/ML App House Price/House Price Prediction/house_price_reg.pkl', 'rb') as file:
        data_house_price = pickle.load(file)
     

    return data_house_price

data_house_price = load_model()

model_best = data_house_price["model"]
le_CentralAir = data_house_price["le_CentralAir"]
le_SaleType = data_house_price["le_SaleType"]
le_SaleCondition = data_house_price["le_SaleCondition"]

def show_predict_page():

    st.title("House Price Predictor")

    st.write("""### Fill up the below information to predict the house price """)

    CentralAir = ["Y", "N"]
    MoSold = [month for month in range(1,13)]
    YrSold = [2006,2007,2008,2009,2010]
    SaleType = ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth']
    SaleCondition = ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']
    
    

    CentralAir.insert(0, "Select Centralised Air Condition")
    MoSold.insert(0, "Select Selling Month")
    YrSold.insert(0, "Select Selling Year")
    SaleType.insert(0, "Select Type of Sale")
    SaleCondition.insert(0, "Select Sale Condition")
    

    CentralAir = st.selectbox("Centralised Air conditioned", CentralAir)
    MoSold = st.selectbox("Selling Month", MoSold)
    YrSold = st.selectbox("Selling Year", YrSold)
    SaleType = st.selectbox("Sale Type", SaleType)
    SaleCondition = st.selectbox("Sale Condition", SaleCondition)
    


    ok = st.button("Show Predicted Price")

    if ok:
        X = np.array([[CentralAir,MoSold,YrSold,SaleType,SaleCondition,]])
        X[: ,0] = le_CentralAir.transform(X[: ,0])
        X[: ,3] = le_SaleType.transform(X[:,3])
        X[: ,4] = le_SaleCondition.transform(X[:,4])
        X = X.astype(float)
        X
        
        y_pred = model_best.predict(X)
        y_pred[0] = y_pred[0].astype(int)

        st.subheader(f"Price : ${y_pred[0]} ")


        # if store_pred[0] == 0:
        #     st.subheader("The store perforance is Bad ☹️")
        # if store_pred[0] ==  1:
        #     st.subheader("The store perforance is Good 😃")
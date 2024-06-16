import pickle
import streamlit as st
import sklearn

st.header('Cars 24 Price Prediction App')

col1, col2, col3 = st.columns(3)

with col1:
    fuel_type = st.selectbox("Select Fuel type :", ("Petrol", "Diesel", "CNG", "Electric", "LPG"))

with col2:
    seller_type = st.selectbox("Select the seller type : ", ("Individual", "Dealer"))

with col3:
    year = st.selectbox("Select the year :", range(1950, 2024))

col4, col5 = st.columns(2)

with col4:
    transmission_type = st.selectbox("Select the Transmission Type", ("Manual", "Automatic"))

with col5:
    seats = st.selectbox("Select the number of seats", [4,5,6,7,8])

col6, col7 = st.columns(2)

with col6:
    mileage = st.slider("Set the Mileage :", 10.0, 30.0, 0.5)

with col7:
    engine = st.slider("Set the Engine Power :", 500, 5000, 100)

col8, col9 = st.columns(2)

with col8:
    km_driven = st.slider("Set the Kms driven :", 0, 200000, 1000)

with col9:
    max_power = st.slider("Set the max power :", 20, 1000, 1)


encode_dict = {
    "fuel_type": {"Deisel" : 1, "Petrol": 2, "CNG": 3, "Electric": 4, "LPG": 5},
    "transmission_type" : {"Manual": 1, "Automatic": 2},
    "seller_type" : {"Individual": 1, "Dealer": 2}
}

def model_pred(year, seller_encode, km_driven, fuel_encode, transmission_encode, mileage, engine, max_power, seats):
    with open("car_pred", "rb") as file:
        reg_model = pickle.load(file)
        input_features = [[year, seller_encode, km_driven, fuel_encode, transmission_encode, mileage, engine, max_power, seats]]
        return reg_model.predict(input_features)


if st.button("Predict"):
    fuel_encode = encode_dict["fuel_type"][fuel_type]
    transmission_encode = encode_dict["transmission_type"][transmission_type]
    seller_encode = encode_dict["seller_type"][seller_type]

    price = model_pred(year, seller_encode, km_driven, fuel_encode, transmission_encode, mileage, engine, max_power, seats)
    st.text("The estimated price of the car is : " + str(price))


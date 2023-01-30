import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
import streamlit as st
#!pip install streamlit

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def quick_missing_imp(data, num_method="median", cat_length=20, target="SATISFACTION"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


def airplane_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, cat_th=4)

    for row in cat_cols:
        dataframe[row] = dataframe[row].str.upper()

    dataframe.drop(["UNNAMED: 0", "ID"], axis=1, inplace=True)

    dataframe["SATISFACTION"] = dataframe["SATISFACTION"].apply(lambda x: 1 if x == "SATISFIED" else 0)

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, cat_th=10)

    for col in num_cols:
        print(col, check_outlier(dataframe, col))

    dataframe.columns = dataframe.columns.str.replace(" ", "_")
    dataframe.columns = dataframe.columns.str.replace("-", "_")
    dataframe.columns = dataframe.columns.str.replace("/", "_")

    dataframe = quick_missing_imp(dataframe, num_method="median", cat_length=20)

    dataframe["GENDER"] = dataframe["GENDER"].apply(lambda x: 1 if x == "FEMALE" else 0)
    dataframe["CUSTOMER_TYPE"] = dataframe["CUSTOMER_TYPE"].apply(lambda x: 1 if x == "LOYAL CUSTOMER" else 0)
    dataframe["TYPE_OF_TRAVEL"] = dataframe["TYPE_OF_TRAVEL"].apply(lambda x: 1 if x == "BUSINESS TRAVEL" else 0)
    dataframe["CLASS"] = dataframe["CLASS"].apply(lambda x: 2 if x == "BUSINESS" else 1 if x == "ECO PLUS" else 0)

    # wifi kullanan kullanmayan
    dataframe["WIFI_AVAIBLE"] = dataframe["INFLIGHT_WIFI_SERVICE"].apply(lambda x: 0 if x == 0 else 1)

    # zengin kiz fakir oglan :RICH_POOR
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["CLASS"] == 2), "RICH_POOR"] = 5
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["CLASS"] == 1), "RICH_POOR"] = 4
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["CLASS"] == 0), "RICH_POOR"] = 3
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["CLASS"] == 2), "RICH_POOR"] = 2
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["CLASS"] == 1), "RICH_POOR"] = 1
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["CLASS"] == 0), "RICH_POOR"] = 0

    # cinsiyete göre uçuş tipi BUSINESS_PERSON
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["TYPE_OF_TRAVEL"] == 1), "BUSINESS_PERSON"] = 3
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["TYPE_OF_TRAVEL"] == 1), "BUSINESS_PERSON"] = 2
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["TYPE_OF_TRAVEL"] == 0), "BUSINESS_PERSON"] = 1
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["TYPE_OF_TRAVEL"] == 0), "BUSINESS_PERSON"] = 0

    # cinsiyete bağlı sadakat IN_LOYAL_GENDER
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["CUSTOMER_TYPE"] == 1), "IN_LOYAL_GENDER"] = 3
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["CUSTOMER_TYPE"] == 0), "IN_LOYAL_GENDER"] = 2
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["CUSTOMER_TYPE"] == 0), "IN_LOYAL_GENDER"] = 1
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["CUSTOMER_TYPE"] == 1), "IN_LOYAL_GENDER"] = 0

    # TYPE_BOARDING
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["ONLINE_BOARDING"] == 5), "TYPE_BOARDING"] = 12
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["ONLINE_BOARDING"] == 5), "TYPE_BOARDING"] = 11
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["ONLINE_BOARDING"] == 4), "TYPE_BOARDING"] = 10
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["ONLINE_BOARDING"] == 4), "TYPE_BOARDING"] = 9
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["ONLINE_BOARDING"] == 3), "TYPE_BOARDING"] = 8
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["ONLINE_BOARDING"] == 3), "TYPE_BOARDING"] = 7
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["ONLINE_BOARDING"] == 2), "TYPE_BOARDING"] = 6
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["ONLINE_BOARDING"] == 2), "TYPE_BOARDING"] = 5
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["ONLINE_BOARDING"] == 1), "TYPE_BOARDING"] = 4
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["ONLINE_BOARDING"] == 1), "TYPE_BOARDING"] = 3
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["ONLINE_BOARDING"] == 0), "TYPE_BOARDING"] = 2
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["ONLINE_BOARDING"] == 0), "TYPE_BOARDING"] = 1

    # classa göre wifi ulaşım WIFI_CLAS
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 1) & (dataframe["CLASS"] == 2), "WIFI_CLAS"] = 6
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 1) & (dataframe["CLASS"] == 1), "WIFI_CLAS"] = 5
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 1) & (dataframe["CLASS"] == 0), "WIFI_CLAS"] = 4
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 0) & (dataframe["CLASS"] == 2), "WIFI_CLAS"] = 3
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 0) & (dataframe["CLASS"] == 1), "WIFI_CLAS"] = 2
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 0) & (dataframe["CLASS"] == 0), "WIFI_CLAS"] = 1

    # TYPE_COMFORT
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["SEAT_COMFORT"] == 5), "TYPE_COMFORT"] = 12
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["SEAT_COMFORT"] == 5), "TYPE_COMFORT"] = 11
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["SEAT_COMFORT"] == 4), "TYPE_COMFORT"] = 10
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["SEAT_COMFORT"] == 4), "TYPE_COMFORT"] = 9
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["SEAT_COMFORT"] == 3), "TYPE_COMFORT"] = 8
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["SEAT_COMFORT"] == 3), "TYPE_COMFORT"] = 7
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["SEAT_COMFORT"] == 2), "TYPE_COMFORT"] = 6
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["SEAT_COMFORT"] == 2), "TYPE_COMFORT"] = 5
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["SEAT_COMFORT"] == 1), "TYPE_COMFORT"] = 4
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["SEAT_COMFORT"] == 1), "TYPE_COMFORT"] = 3
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 1) & (dataframe["SEAT_COMFORT"] == 0), "TYPE_COMFORT"] = 2
    dataframe.loc[(dataframe["TYPE_OF_TRAVEL"] == 0) & (dataframe["SEAT_COMFORT"] == 0), "TYPE_COMFORT"] = 1

    # sadakate göre müşteri sınıflandırılması; LOY_CLASS"
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 1) & (dataframe["CLASS"] == 2), "LOY_CLASS"] = 6
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 1) & (dataframe["CLASS"] == 1), "LOY_CLASS"] = 5
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 1) & (dataframe["CLASS"] == 0), "LOY_CLASS"] = 4
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 0) & (dataframe["CLASS"] == 2), "LOY_CLASS"] = 3
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 0) & (dataframe["CLASS"] == 1), "LOY_CLASS"] = 2
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 0) & (dataframe["CLASS"] == 0), "LOY_CLASS"] = 1

    # sadakete göre yolculuk sınıflandırılması; LOY_TRAVEL
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 1) & (dataframe["TYPE_OF_TRAVEL"] == 1), "LOY_TRAVEL"] = 4
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 1) & (dataframe["TYPE_OF_TRAVEL"] == 0), "LOY_TRAVEL"] = 3
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 0) & (dataframe["TYPE_OF_TRAVEL"] == 1), "LOY_TRAVEL"] = 2
    dataframe.loc[(dataframe["CUSTOMER_TYPE"] == 0) & (dataframe["TYPE_OF_TRAVEL"] == 0), "LOY_TRAVEL"] = 1

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, cat_th=20)

    dataframe['FLIGHT_DISTANCE'] = np.log1p(dataframe['FLIGHT_DISTANCE'])
    dataframe['DEPARTURE_DELAY_IN_MINUTES'] = np.log1p(dataframe['DEPARTURE_DELAY_IN_MINUTES'])
    dataframe['ARRIVAL_DELAY_IN_MINUTES'] = np.log1p(dataframe['ARRIVAL_DELAY_IN_MINUTES'])

    cat_cols = [col for col in cat_cols if "SATISFACTION" not in col]

    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    y = dataframe["SATISFACTION"]
    X = dataframe.drop(["SATISFACTION"], axis=1)

    return X, y



new_model = joblib.load(r"D:\pythonProject\ada_clf.pkl")


image = Image.open(r"C:\Users\erenk\Desktop\Logo Data Crew.jpg")

st.image(image, caption='Presented By : Data Crew')


st.title("Are They Satisfied")


GENDER = st.number_input("Cinsiyetinizi Seçiniz | Kadın : 1 | Erkek : 0", 0, 1)
CUSTOMER_TYPE = st.number_input("Müşteri tipini seçiniz: Sürekli Müşteri ise : 1 | Geçici Müşteri ise : 0", 0, 1)
AGE = st.number_input("Yaş Girin", 1, 90,)
TYPE_OF_TRAVEL = st.number_input("Ucuş Tipi -- Business : 1 | Personal : 0 ", 0, 1)
CLASS = st.number_input("Uçuş Sınıfı -- Business : 2 | Eco Plus : 1 | Eco : 0", 0, 2)
FLIGHT_DISTANCE = st.number_input("Uçuş Mesafesi Giriniz", 1, 5000)
INFLIGHT_WIFI_SERVICE = st.number_input("Wifi Servisi Puanlayınız | 0-5", 0, 5)
DEPARTURE_ARRIVAL_TIME_CONVENIENT = st.number_input("Zamanında Kalkış-İniş Durumunu Puanlayınız | 0-5", 0, 5)
EASE_OF_ONLINE_BOOKING = st.number_input("Online Rezarvasyonu Puanlayınız | 0-5", 0, 5)
GATE_LOCATION = st.number_input("Kapı Konumunu Puanlayınız | 0-5", 0, 5)
FOOD_AND_DRINK = st.number_input("Yiyecek İçecek Servisini Puanlayınız | 0-5", 0, 5)
ONLINE_BOARDING = st.number_input("Online Chek-İn Servisini Puanlayınız | 0-5", 0, 5)
SEAT_COMFORT = st.number_input("Koltuk Konforunu Puanlayınız | 0-5", 0, 5)
INFLIGHT_ENTERTAINMENT = st.number_input("Uçak İçi Eğlence Durumunu Puanlayınız | 0-5", 0, 5)
ON_BOARD_SERVICE = st.number_input("Yerleşik Hizmet Servisini Puanlayınız | 0-5", 0, 5)
LEG_ROOM_SERVICE = st.number_input("Ayak Mesafesini Puanlayınız | 0-5", 0, 5)
BAGGAGE_HANDLING = st.number_input("Bagaj Taşıma Servisini Puanlayınız | 0-5", 0, 5)
CHECKIN_SERVICE = st.number_input("Check-in Kolaylığı Durumunu Puanlayınız | 0-5", 0, 5)
INFLIGHT_SERVICE = st.number_input("Uçak İçi Hizmet Durumunu Puanlayınız | 0-5", 0, 5)
CLEANLINESS = st.number_input("Temizlik Durumunu Puanlayınız | 0-5", 0, 5)
DEPARTURE_DELAY_IN_MINUTES = st.number_input("Kalkış Gecikmesi Zamanını Belirtiniz", 1, 2000)
ARRIVAL_DELAY_IN_MINUTES = st.number_input("Varış Gecikmesi Zamanını Belirtiniz", 1, 2000)
WIFI_AVAIBLE = st.number_input("Wifi Hizmeti Var : 1 | Wifi Hizmeti Yok : 0", 0, 1)
RICH_POOR = st.number_input("Uçuş Tipiniz ve Class Durumunuzu Belirtiniz ", 0, 5)
BUSINESS_PERSON = st.number_input("İş İnsanı: Erkek: 0 Bus: 1 ise 3 | Kadın:1 Bus:1 ise 2", 0, 3)
IN_LOYAL_GENDER = st.number_input("Sürekli Müşteri ve Cinsiyet: Sürekli:1 ve Erkek:0 ise 3", 0, 3)
TYPE_BOARDING = st.number_input("Biniş Şekli: Bussines:1 ve Boarding: 5 ise 12", 1, 12)
WIFI_CLAS = st.number_input("Wifi Class", 1, 6)
TYPE_COMFORT = st.number_input("Tip Konforu: Bussines:1 ve Boarding: 5 ise 12", 1, 12)
LOY_CLASS = st.number_input("Süreklilik ve Yolcu Sınıfı: Sürekli:1 Class:2 ise 6", 1, 6)
LOY_TRAVEL = st.number_input("Süreklilik ve Uçuş Tipi", 1, 4)


def predict():
    row = np.array([GENDER, CUSTOMER_TYPE, AGE, TYPE_OF_TRAVEL,
       CLASS, FLIGHT_DISTANCE, INFLIGHT_WIFI_SERVICE,
       DEPARTURE_ARRIVAL_TIME_CONVENIENT, EASE_OF_ONLINE_BOOKING,
       GATE_LOCATION, FOOD_AND_DRINK, ONLINE_BOARDING, SEAT_COMFORT,
       INFLIGHT_ENTERTAINMENT, ON_BOARD_SERVICE, LEG_ROOM_SERVICE,
       BAGGAGE_HANDLING, CHECKIN_SERVICE, INFLIGHT_SERVICE,
       CLEANLINESS, DEPARTURE_DELAY_IN_MINUTES, ARRIVAL_DELAY_IN_MINUTES, WIFI_AVAIBLE,
       RICH_POOR, BUSINESS_PERSON, IN_LOYAL_GENDER,TYPE_BOARDING, WIFI_CLAS, TYPE_COMFORT, LOY_CLASS, LOY_TRAVEL])
    X = pd.DataFrame([row], columns=columns)
    prediction = new_model.predict(X)[0]

    if prediction == 1:
        st.success("Memnun Müşteri :thumbsup:")
    else:
        st.error("Memnun Olmayan Müşteri :thumbsdown:")

st.button("Predict", on_click=predict)


columns = ['GENDER', 'CUSTOMER_TYPE', 'AGE', 'TYPE_OF_TRAVEL',
       'CLASS', 'FLIGHT_DISTANCE', 'INFLIGHT_WIFI_SERVICE',
       'DEPARTURE_ARRIVAL_TIME_CONVENIENT', 'EASE_OF_ONLINE_BOOKING',
       'GATE_LOCATION', 'FOOD_AND_DRINK', 'ONLINE_BOARDING', 'SEAT_COMFORT',
       'INFLIGHT_ENTERTAINMENT', 'ON_BOARD_SERVICE', 'LEG_ROOM_SERVICE',
       'BAGGAGE_HANDLING', 'CHECKIN_SERVICE', 'INFLIGHT_SERVICE',
       'CLEANLINESS', 'DEPARTURE_DELAY_IN_MINUTES', 'ARRIVAL_DELAY_IN_MINUTES',"WIFI_AVAIBLE",
       "RICH_POOR","BUSINESS_PERSON", "IN_LOYAL_GENDER","TYPE_BOARDING", "WIFI_CLAS", "TYPE_COMFORT", "LOY_CLASS", "LOY_TRAVEL"]


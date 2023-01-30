################################################################
# AIRPLANE SATISFACTION PREDICT
################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import joblib
import missingno as msno
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, cross_val_score
from catboost import CatBoostClassifier
warnings.simplefilter(action="ignore")

################################################################
# Ayarlar
################################################################

df = pd.read_csv(r"C:\Users\erenk\Desktop\train.csv\train.csv")


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

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

"""
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

    dataframe = quick_missing_imp(dataframe, num_method="median", cat_length=20)

    dataframe["GENDER"] = dataframe["GENDER"].apply(lambda x: 1 if x == "FEMALE" else 0)
    dataframe["CUSTOMER TYPE"] = dataframe["CUSTOMER TYPE"].apply(lambda x: 1 if x == "LOYAL CUSTOMER" else 0)
    dataframe["TYPE OF TRAVEL"] = dataframe["TYPE OF TRAVEL"].apply(lambda x: 1 if x == "BUSINESS TRAVEL" else 0)
    dataframe["CLASS"] = dataframe["CLASS"].apply(lambda x: 2 if x == "BUSINESS" else 1 if x == "ECO PLUS" else 0)

    # wifi kullanan kullanmayan
    dataframe["WIFI_AVAIBLE"] = dataframe["INFLIGHT WIFI SERVICE"].apply(lambda x: 0 if x == 0 else 1)

    # zengin kiz fakir oglan :RICH_POOR
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["CLASS"] == 2), "RICH_POOR"] = 5
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["CLASS"] == 1), "RICH_POOR"] = 4
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["CLASS"] == 0), "RICH_POOR"] = 3
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["CLASS"] == 2), "RICH_POOR"] = 2
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["CLASS"] == 1), "RICH_POOR"] = 1
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["CLASS"] == 0), "RICH_POOR"] = 0

    # cinsiyete göre uçuş tipi BUSINESS_PERSON
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["TYPE OF TRAVEL"] == 1), "BUSINESS_PERSON"] = 3
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["TYPE OF TRAVEL"] == 1), "BUSINESS_PERSON"] = 2
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["TYPE OF TRAVEL"] == 0), "BUSINESS_PERSON"] = 1
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["TYPE OF TRAVEL"] == 0), "BUSINESS_PERSON"] = 0

    # cinsiyete bağlı sadakat IN_LOYAL_GENDER
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["CUSTOMER TYPE"] == 1), "IN_LOYAL_GENDER"] = 3
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["CUSTOMER TYPE"] == 0), "IN_LOYAL_GENDER"] = 2
    dataframe.loc[(dataframe["GENDER"] == 0) & (dataframe["CUSTOMER TYPE"] == 0), "IN_LOYAL_GENDER"] = 1
    dataframe.loc[(dataframe["GENDER"] == 1) & (dataframe["CUSTOMER TYPE"] == 1), "IN_LOYAL_GENDER"] = 0

    # TYPE_BOARDING
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["ONLINE BOARDING"] == 5), "TYPE_BOARDING"] = 12
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["ONLINE BOARDING"] == 5), "TYPE_BOARDING"] = 11
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["ONLINE BOARDING"] == 4), "TYPE_BOARDING"] = 10
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["ONLINE BOARDING"] == 4), "TYPE_BOARDING"] = 9
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["ONLINE BOARDING"] == 3), "TYPE_BOARDING"] = 8
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["ONLINE BOARDING"] == 3), "TYPE_BOARDING"] = 7
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["ONLINE BOARDING"] == 2), "TYPE_BOARDING"] = 6
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["ONLINE BOARDING"] == 2), "TYPE_BOARDING"] = 5
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["ONLINE BOARDING"] == 1), "TYPE_BOARDING"] = 4
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["ONLINE BOARDING"] == 1), "TYPE_BOARDING"] = 3
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["ONLINE BOARDING"] == 0), "TYPE_BOARDING"] = 2
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["ONLINE BOARDING"] == 0), "TYPE_BOARDING"] = 1

    # classa göre wifi ulaşım WIFI_CLAS
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 1) & (dataframe["CLASS"] == 2), "WIFI_CLAS"] = 6
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 1) & (dataframe["CLASS"] == 1), "WIFI_CLAS"] = 5
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 1) & (dataframe["CLASS"] == 0), "WIFI_CLAS"] = 4
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 0) & (dataframe["CLASS"] == 2), "WIFI_CLAS"] = 3
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 0) & (dataframe["CLASS"] == 1), "WIFI_CLAS"] = 2
    dataframe.loc[(dataframe["WIFI_AVAIBLE"] == 0) & (dataframe["CLASS"] == 0), "WIFI_CLAS"] = 1

    # TYPE_COMFORT
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["SEAT COMFORT"] == 5), "TYPE_COMFORT"] = 12
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["SEAT COMFORT"] == 5), "TYPE_COMFORT"] = 11
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["SEAT COMFORT"] == 4), "TYPE_COMFORT"] = 10
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["SEAT COMFORT"] == 4), "TYPE_COMFORT"] = 9
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["SEAT COMFORT"] == 3), "TYPE_COMFORT"] = 8
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["SEAT COMFORT"] == 3), "TYPE_COMFORT"] = 7
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["SEAT COMFORT"] == 2), "TYPE_COMFORT"] = 6
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["SEAT COMFORT"] == 2), "TYPE_COMFORT"] = 5
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["SEAT COMFORT"] == 1), "TYPE_COMFORT"] = 4
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["SEAT COMFORT"] == 1), "TYPE_COMFORT"] = 3
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 1) & (dataframe["SEAT COMFORT"] == 0), "TYPE_COMFORT"] = 2
    dataframe.loc[(dataframe["TYPE OF TRAVEL"] == 0) & (dataframe["SEAT COMFORT"] == 0), "TYPE_COMFORT"] = 1

    # sadakate göre müşteri sınıflandırılması; LOY_CLASS"
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 1) & (dataframe["CLASS"] == 2), "LOY_CLASS"] = 6
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 1) & (dataframe["CLASS"] == 1), "LOY_CLASS"] = 5
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 1) & (dataframe["CLASS"] == 0), "LOY_CLASS"] = 4
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 0) & (dataframe["CLASS"] == 2), "LOY_CLASS"] = 3
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 0) & (dataframe["CLASS"] == 1), "LOY_CLASS"] = 2
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 0) & (dataframe["CLASS"] == 0), "LOY_CLASS"] = 1

    # sadakete göre yolculuk sınıflandırılması; LOY_TRAVEL
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 1) & (dataframe["TYPE OF TRAVEL"] == 1), "LOY_TRAVEL"] = 4
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 1) & (dataframe["TYPE OF TRAVEL"] == 0), "LOY_TRAVEL"] = 3
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 0) & (dataframe["TYPE OF TRAVEL"] == 1), "LOY_TRAVEL"] = 2
    dataframe.loc[(dataframe["CUSTOMER TYPE"] == 0) & (dataframe["TYPE OF TRAVEL"] == 0), "LOY_TRAVEL"] = 1

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, cat_th=20)

    dataframe['FLIGHT DISTANCE'] = np.log1p(dataframe['FLIGHT DISTANCE'])
    dataframe['DEPARTURE DELAY IN MINUTES'] = np.log1p(dataframe['DEPARTURE DELAY IN MINUTES'])
    dataframe['ARRIVAL DELAY IN MINUTES'] = np.log1p(dataframe['ARRIVAL DELAY IN MINUTES'])

    cat_cols = [col for col in cat_cols if "SATISFACTION" not in col]

    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    y = dataframe["SATISFACTION"]
    X = dataframe.drop(["SATISFACTION"], axis=1)

    return X, y
"""

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

def base_models_score(X, y, scoring="score"):
    print("Base Models Score....")
    models = [#("LR", LogisticRegression()),
              #("KNN", KNeighborsClassifier()),
              #("CART", DecisionTreeClassifier()),
              #("RF", RandomForestClassifier()),
              ("Adaboost", AdaBoostClassifier()),
              #("GBM", GradientBoostingClassifier()),
              #("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
              #("CatBoost", CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]
    for name, model in models:
        print(name)
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cvs = cross_val_score(model, X, y, scoring=score, cv=5).mean()
            print(score + " score:" + str(cvs))

def adaboost_predict_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=46)

    base_models_score(X_train, y_train)

    AdaBoost_model = AdaBoostClassifier(random_state=17).fit(X_train, y_train)

    ada_model_param = {"n_estimators": [250, 300, 400],
                       'learning_rate': [0.01, 0.1, 1.0]}
    AdaBoost_best_grid = GridSearchCV(AdaBoost_model, ada_model_param, cv=10, n_jobs=-1, verbose=False).fit(X_train,
                                                                                                            y_train)
    ada_final_model = AdaBoost_model.set_params(**AdaBoost_best_grid.best_params_)

    return ada_final_model

def main():
    df = pd.read_csv(r"C:\Users\erenk\Desktop\train.csv\train.csv")
    X, y = airplane_data_prep(df)
    ada_final_df = adaboost_predict_model(X, y)
    joblib.dump(ada_final_df, "../ada_clf.pkl")
    return ada_final_df

################################################################
if __name__ == "__main__":
    print("PROCESS STARTED....")
    main()
################################################################

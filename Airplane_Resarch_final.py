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
from sklearn.metrics import r2_score


warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

################################################################

df = pd.read_csv(r"C:\Users\erenk\Desktop\train.csv\train.csv")

################################################################
# 1. Exploratory Data Analysis
################################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(data):
    corr = data.corr().round(2)

    # Mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set figure size
    f, ax = plt.subplots(figsize=(20, 20))

    # Define custom colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap
    corr=sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.tight_layout()
    return corr

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

check_df(df)

################################################################
# Degiskenler ve Kolonların Büyütülmesi ve Çıkarılması
################################################################

df.columns = [col.upper() for col in df.columns]

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=4)

for row in cat_cols:
    df[row] = df[row].str.upper()

df.drop(["UNNAMED: 0", "ID"], axis=1, inplace=True)

################################################################
# Bagımlı degiskenin değerlerinin sayısallastırılması
################################################################

df["SATISFACTION"] = df["SATISFACTION"].apply(lambda x: 1 if x == "SATISFIED" else 0)

################################################################
# Değişken türlerinin ayrıştırılması
################################################################

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=10)

################################################################
# BAGIMLI DEGISKENIN DAGILIM GORSELLESTIRMESI
################################################################

plt.pie(df["SATISFACTION"].value_counts(), labels = ["Neutral or dissatisfied", "Satisfied"],
        colors=sns.color_palette("rocket_r"), autopct='%1.1f%%')
plt.show(blok=True)

################################################################
# NUMERIK DEGISKEN ANALIZI
################################################################

for col in num_cols:
    num_summary(df, col, plot=True)

################################################################
# NUMERIK DEGISKENIN TARGET ANALIZI
################################################################

for col in num_cols:
    target_summary_with_num(df, "SATISFACTION", col)

################################################################
# GORSELLESTIRME TARGET
################################################################

for col in num_cols:
    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x="SATISFACTION", y=col, palette="YlOrBr", data=df, ax=ax[0])
    sns.histplot(df, x=col, hue="SATISFACTION", multiple="stack", palette="YlOrBr", edgecolor=".3", linewidth=.5, ax=ax[1])
    plt.show()

################################################################
# KATEGORIK DEGISKENIN VERIDE DAGILIMI
################################################################

for col in cat_cols:
    cat_summary(df, col, plot=True)

################################################################
# KATEGORIK DEGISKENIN TARGET ANALIZI
################################################################

for col in cat_cols:
    target_summary_with_cat(df, "SATISFACTION", col)

################################################################
# KATEGORIK DEGISKENIN TARGET ANALIZI GORSEL
################################################################

for col in cat_cols:
    sns.catplot("SATISFACTION", col=col, col_wrap=2, data=df, kind="count", height=2.5, aspect=1.5)
    plt.show()

################################################################
# KORELASYON MATRIX
################################################################

correlation_matrix(df)
plt.show()

################################################################
# 2. Data Preprocessing & Feature Engineering
################################################################

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

################################################################
# AYKIRI DEGERLERIN YAKALANMASI
################################################################

for col in num_cols:
    print(col, check_outlier(df, col))

################################################################
# BOS DEGERLERIN MEDIAN ILE DOLDURULMASI
################################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

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

df = quick_missing_imp(df, num_method="median", cat_length=20)

################################################################
# DEGISKENLERIN MANUEL OLARAK SAYISALLASTIRILMASI
################################################################

df['GENDER'] = df['GENDER'].apply(lambda x: 1 if x == 'FEMALE' else 0)
df['CUSTOMER TYPE'] = df['CUSTOMER TYPE'].apply(lambda x: 1 if x == 'LOYAL CUSTOMER' else 0)
df['TYPE OF TRAVEL'] = df['TYPE OF TRAVEL'].apply(lambda x: 1 if x == 'BUSINESS TRAVEL' else 0)
df['CLASS'] = df['CLASS'].apply(lambda x: 2 if x == 'BUSINESS' else 1 if x == 'ECO PLUS' else 0)

################################################################
# YENI DEGISKENLERIN URETILMESI
################################################################


# zengin kız fakir oglan :RICH_POOR
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["CLASS"] == 2), "RICH_POOR"] = 5
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["CLASS"] == 1), "RICH_POOR"] = 4
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["CLASS"] == 0), "RICH_POOR"] = 3
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["CLASS"] == 2), "RICH_POOR"] = 2
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["CLASS"] == 1), "RICH_POOR"] = 1
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["CLASS"] == 0), "RICH_POOR"] = 0

# cinsiyete göre uçuş tipi BUSINESS_PERSON
df.loc[(df["GENDER"] == 0) & (df["TYPE OF TRAVEL"] == 1), "BUSINESS_PERSON"] = 3
df.loc[(df["GENDER"] == 1) & (df["TYPE OF TRAVEL"] == 1), "BUSINESS_PERSON"] = 2
df.loc[(df["GENDER"] == 1) & (df["TYPE OF TRAVEL"] == 0), "BUSINESS_PERSON"] = 1
df.loc[(df["GENDER"] == 0) & (df["TYPE OF TRAVEL"] == 0), "BUSINESS_PERSON"] = 0

# cinsiyete bağlı sadakat IN_LOYAL_GENDER
df.loc[(df["GENDER"] == 0) & (df["CUSTOMER TYPE"] == 1), "IN_LOYAL_GENDER"] = 3
df.loc[(df["GENDER"] == 1) & (df["CUSTOMER TYPE"] == 0), "IN_LOYAL_GENDER"] = 2
df.loc[(df["GENDER"] == 0) & (df["CUSTOMER TYPE"] == 0), "IN_LOYAL_GENDER"] = 1
df.loc[(df["GENDER"] == 1) & (df["CUSTOMER TYPE"] == 1), "IN_LOYAL_GENDER"] = 0

# TYPE_BOARDING
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["ONLINE BOARDING"] == 5), "TYPE_BOARDING"] = 12
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["ONLINE BOARDING"] == 5), "TYPE_BOARDING"] = 11
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["ONLINE BOARDING"] == 4), "TYPE_BOARDING"] = 10
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["ONLINE BOARDING"] == 4), "TYPE_BOARDING"] = 9
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["ONLINE BOARDING"] == 3), "TYPE_BOARDING"] = 8
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["ONLINE BOARDING"] == 3), "TYPE_BOARDING"] = 7
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["ONLINE BOARDING"] == 2), "TYPE_BOARDING"] = 6
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["ONLINE BOARDING"] == 2), "TYPE_BOARDING"] = 5
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["ONLINE BOARDING"] == 1), "TYPE_BOARDING"] = 4
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["ONLINE BOARDING"] == 1), "TYPE_BOARDING"] = 3
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["ONLINE BOARDING"] == 0), "TYPE_BOARDING"] = 2
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["ONLINE BOARDING"] == 0), "TYPE_BOARDING"] = 1

df["WIFI_AVAIBLE"] = df["INFLIGHT WIFI SERVICE"].apply(lambda x: 0 if x == 0 else 1)

df.loc[(df["WIFI_AVAIBLE"] == 1) & (df["CLASS"] == 2), "WIFI_CLAS"] = 6
df.loc[(df["WIFI_AVAIBLE"] == 1) & (df["CLASS"] == 1), "WIFI_CLAS"] = 5
df.loc[(df["WIFI_AVAIBLE"] == 1) & (df["CLASS"] == 0), "WIFI_CLAS"] = 4
df.loc[(df["WIFI_AVAIBLE"] == 0) & (df["CLASS"] == 2), "WIFI_CLAS"] = 3
df.loc[(df["WIFI_AVAIBLE"] == 0) & (df["CLASS"] == 1), "WIFI_CLAS"] = 2
df.loc[(df["WIFI_AVAIBLE"] == 0) & (df["CLASS"] == 0), "WIFI_CLAS"] = 1

# TYPE_COMFORT
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["SEAT COMFORT"] == 5), "TYPE_COMFORT"] = 12
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["SEAT COMFORT"] == 5), "TYPE_COMFORT"] = 11
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["SEAT COMFORT"] == 4), "TYPE_COMFORT"] = 10
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["SEAT COMFORT"] == 4), "TYPE_COMFORT"] = 9
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["SEAT COMFORT"] == 3), "TYPE_COMFORT"] = 8
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["SEAT COMFORT"] == 3), "TYPE_COMFORT"] = 7
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["SEAT COMFORT"] == 2), "TYPE_COMFORT"] = 6
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["SEAT COMFORT"] == 2), "TYPE_COMFORT"] = 5
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["SEAT COMFORT"] == 1), "TYPE_COMFORT"] = 4
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["SEAT COMFORT"] == 1), "TYPE_COMFORT"] = 3
df.loc[(df["TYPE OF TRAVEL"] == 1) & (df["SEAT COMFORT"] == 0), "TYPE_COMFORT"] = 2
df.loc[(df["TYPE OF TRAVEL"] == 0) & (df["SEAT COMFORT"] == 0), "TYPE_COMFORT"] = 1

# sadakate göre müşteri sınıflandırılması; LOY_CLASS"
df.loc[(df["CUSTOMER TYPE"] == 1) & (df["CLASS"] == 2), "LOY_CLASS"] = 6
df.loc[(df["CUSTOMER TYPE"] == 1) & (df["CLASS"] == 1), "LOY_CLASS"] = 5
df.loc[(df["CUSTOMER TYPE"] == 1) & (df["CLASS"] == 0), "LOY_CLASS"] = 4
df.loc[(df["CUSTOMER TYPE"] == 0) & (df["CLASS"] == 2), "LOY_CLASS"] = 3
df.loc[(df["CUSTOMER TYPE"] == 0) & (df["CLASS"] == 1), "LOY_CLASS"] = 2
df.loc[(df["CUSTOMER TYPE"] == 0) & (df["CLASS"] == 0), "LOY_CLASS"] = 1

# sadakete göre yolculuk sınıflandırılması; LOY_TRAVEL
df.loc[(df["CUSTOMER TYPE"] == 1) & (df["TYPE OF TRAVEL"] == 1), "LOY_TRAVEL"] = 4
df.loc[(df["CUSTOMER TYPE"] == 1) & (df["TYPE OF TRAVEL"] == 0), "LOY_TRAVEL"] = 3
df.loc[(df["CUSTOMER TYPE"] == 0) & (df["TYPE OF TRAVEL"] == 1), "LOY_TRAVEL"] = 2
df.loc[(df["CUSTOMER TYPE"] == 0) & (df["TYPE OF TRAVEL"] == 0), "LOY_TRAVEL"] = 1

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=20)

################################################################
################################################################

################################################################
# CARPIKLIGIN KONTROL EDİLMESİ VE DUZENLENMESI
################################################################

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(df, 'FLIGHT DISTANCE')
plt.subplot(6, 1, 2)
check_skew(df, 'DEPARTURE DELAY IN MINUTES')
plt.subplot(6, 1, 3)
check_skew(df, 'ARRIVAL DELAY IN MINUTES')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

#################################################################
# NORMAL DAGILIMIN SAGLANMASI ICIN LOG TRANSFORMATION UYGULANMASI
#################################################################

df['FLIGHT DISTANCE'] = np.log1p(df['FLIGHT DISTANCE'])
df['DEPARTURE DELAY IN MINUTES'] = np.log1p(df['DEPARTURE DELAY IN MINUTES'])
df['ARRIVAL DELAY IN MINUTES'] = np.log1p(df['ARRIVAL DELAY IN MINUTES'])

################################################################
# BAGIMLI DEGISKENIN BELIRLENMESI
################################################################

cat_cols = [col for col in cat_cols if "SATISFACTION" not in col]

y = df["SATISFACTION"]
X = df.drop(["SATISFACTION"], axis=1)

################################################################
# STANDARTLASTIRMA
################################################################

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
df.head()


################################################################
# DATA PREP FONK.
################################################################

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

df = pd.read_csv(r"C:\Users\erenk\Desktop\train.csv\train.csv")

X, y = airplane_data_prep(df)


################################################################
# 3. Base Models
################################################################

def base_models_score(X, y, scoring="score"):
    print("Base Models Score....")
    models = [("LR", LogisticRegression()),
              ("KNN", KNeighborsClassifier()),
              ("CART", DecisionTreeClassifier()),
              ("RF", RandomForestClassifier()),
              ("Adaboost", AdaBoostClassifier()),
              ("GBM", GradientBoostingClassifier()),
              ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
              ("CatBoost", CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]
    for name, model in models:
        print(name)
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cvs = cross_val_score(model, X, y, scoring=score, cv=5).mean()
            print(score + " score:" + str(cvs))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=46)

base_models_score(X_train, y_train)


################################################################
# 4. Automated Hyperparameter Optimization
################################################################
"""
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300, 500, 700]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 10],
                  "n_estimators": [100, 200, 300, 400]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 700, 900]}

classifiers = [#('KNN', KNeighborsClassifier(), knn_params),
               #("CART", DecisionTreeClassifier(), cart_params),
               #("RF", RandomForestClassifier(), rf_params),
               #('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=5, score=["roc_auc"]):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")

    for scoring in score:
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)


        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
    best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y, cv=5, score=["roc_auc", "f1"])
"""

################################################################
# Modeling
################################################################

# x ve y data_prep fpn Kgeliyor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=46)

################################################################
# adaboost
################################################################

AdaBoost_model = AdaBoostClassifier(random_state=17).fit(X_train, y_train)
AdaBoost_model.get_params()

ada_model_param = {"n_estimators": [250, 300, 400],
                   'learning_rate': [0.01, 0.1, 1.0]}
AdaBoost_best_grid = GridSearchCV(AdaBoost_model, ada_model_param, cv=10, n_jobs=-1, verbose=False).fit(X_train, y_train)
ada_final_model = AdaBoost_model.set_params(**AdaBoost_best_grid.best_params_)

y_pred = ada_final_model.predict(X_train)

ada_final_model.score(X_train, y_train)

r2_score(y_train, y_pred)

# ada_final_model.score(X_train, y_train)
# Out[16]: 0.9281383337076129
# r2 0.7073280539993871

################################################################
# ADABOOST FONK. HALE GETIRILMESI
################################################################

def adaboost_predict_model(X_train, y_train):
    AdaBoost_model = AdaBoostClassifier(random_state=17).fit(X_train, y_train)

    ada_model_param = {"n_estimators": [250, 300, 400],
                       'learning_rate': [0.01, 0.1, 1.0]}
    AdaBoost_best_grid = GridSearchCV(AdaBoost_model, ada_model_param, cv=10, n_jobs=-1, verbose=False).fit(X_train,
                                                                                                            y_train)
    ada_final_model = AdaBoost_model.set_params(**AdaBoost_best_grid.best_params_)
    return ada_final_model

################################################################
# xgboost predict
"""
XgBoost_model = XGBClassifier(random_state=17).fit(X_train, y_train)
XgBoost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [2, 8],
                  "n_estimators": [100, 200]}

XgBoost_best_grid = GridSearchCV(XgBoost_model, XgBoost_params, cv=10, n_jobs=-1, verbose=True).fit(X_train, y_train)
XgBoost_final_model = XgBoost_model.set_params(**XgBoost_best_grid.best_params_)
y_pred = XgBoost_final_model.predict(X_train)
XgBoost_final_model.score(X_train, y_train)
r2_score(y_train, y_pred)
Out[18]: 0.9802380417695935
Out[19]: 0.9195152148498315
"""
################################################################
# Logreg predict
"""
logreg_model = LogisticRegression(random_state=17).fit(X_train, y_train)
logreg_model.get_params()

logreg_model_param = {"max_iter": [200, 250, 300]}
logreg_best_grid = GridSearchCV(logreg_model, logreg_model_param, cv=10, n_jobs=-1).fit(X_train, y_train)
logreg_final_model = logreg_model.set_params(**logreg_best_grid.best_params_)

y_pred = logreg_final_model.predict(X_train)
logreg_final_model.score(X_train, y_train)
# 0.903788778030862
r2_score(y_train, y_pred)
# 0.6081593008679294
"""
################################################################
# CatBoost predict
"""
CatBoost_model = CatBoostClassifier(random_state=17, verbose=False).fit(X_train, y_train)
CatBoost_model.get_params()

CatBoost_model_param = {"iterations": [400, 500, 600],
                        "learning_rate": [0.1, 0.5, 1],
                        "depth": [4, 6, 8]}
CatBoost_best_grid = GridSearchCV(CatBoost_model, CatBoost_model_param, cv=10, n_jobs=10, verbose=False).fit(X_train, y_train)
CatBoost_final_model = CatBoostClassifier(depth=6, iterations=400, learning_rate=0.1, random_state=17, verbose=False).fit(X_train, y_train)

y_pred = CatBoost_final_model.predict(X_train)
CatBoost_final_model.score(X_train, y_train)
# 0.9730839562413782
r2_score(y_train, y_pred)
# 0.8903786773685204
"""
################################################################
# RF predict
"""
rf_models = RandomForestClassifier(random_state=17).fit(X_train, y_train)
rf_models.get_params()
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 400]}

rf_best_grid = GridSearchCV(rf_models, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
y_pred = rf_models.predict(X_train)
rf_models.score(X_train, y_train)
r2_score(y_train, y_pred)
"""
################################################################
# LGBM
"""
lgbm_model = LGBMClassifier(random_state=17).fit(X_train, y_train)
y_pred = lgbm_model.predict(X_train)
lgbm_model.score(X_train, y_train)
r2_score(y_train, y_pred)

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 700, 900]}

lgbm_best_grid = GridSearchCV(lgbm_model, lightgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X_train, y_train)
lgbm_final_model = lgbm_model.set_params(**lgbm_best_grid.bestparams)
y_pred = lgbm_final_model.predict(X_train)
lgbm_final_model.score(X_train, y_train)
r2_score(y_train, y_pred)
"""

################################################################
# predict sonsrası tahmin etme
################################################################

X.columns
random_user = X.sample(1)
ada_final_model.predict(random_user)
df[df.index == 80601]

################################################################
# 6. Feature Importance özellik önemi değişken önemi
################################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(ada_final_model, X, num=10)

################################################################
# test setinı tanıtıp model tahmini yapmaya hazır
################################################################
df_test = pd.read_csv(r"C:\Users\erenk\Desktop\test.csv\test.csv")
X, y = airplane_data_prep(df_test)
################################################################
random_user = X.sample(1)
ada_final_model.predict(random_user)
df[df.index == 95205]
################################################################

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

X, y = airplane_data_prep(df)

ada_final_model = adaboost_predict_model(X, y)

################################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

y_pred = ada_final_model.predict(X)
plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
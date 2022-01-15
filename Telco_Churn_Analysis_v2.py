# Telco Churn Feature Engineering

# Bussiness Problem

# It is desirable to develop a machine learning model that can predict customers who will leave the company.
# You are expected to perform the necessary data analysis and feature engineering steps before developing the model.


# Dataset Story

# Telco churn data includes information about a fictitious telecom company that provided home phone,
# and Internet services to 7,043 California customers in the third quarter.
# It shows which customers have left, stayed or signed up for their service.

# CustomerId : customer id
# Gender : Gender
# SeniorCitizen : Whether the customer is old (1, 0)
# Partner : Whether the customer has a partner (Yes, No)
# Dependents : Whether the customer has dependents (Yes, No)
# tenure : Number of months the customer has stayed with the company
# PhoneService : Whether the customer has telephone service (Yes, No)
# MultipleLines : Whether the customer has more than one line (Yes, No, No phone service)
# InternetService : Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity : Whether the customer has online security (Yes, No, no Internet service)
# OnlineBackup : Whether the customer has an online backup (Yes, No, no Internet service)
# DeviceProtection : Whether the customer has device protection (Yes, No, no Internet service)
# TechSupport : Whether the customer has technical support (Yes, No, no Internet service)
# StreamingTV : Whether the customer has TV broadcast (Yes, No, no Internet service)
# StreamingMovies : Whether the client is streaming movies (Yes, No, no Internet service)
# Contract : Customer's contract duration (Month to month, One year, Two years)
# PaperlessBilling : Whether the customer has a paperless invoice (Yes, No)
# PaymentMethod : Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges : The monthly amount charged to the customer
# TotalCharges : Total amount charged from customer
# Churn : Whether the customer is using (Yes or No)

# Task 1 : Exploring Data Analysis

# Step 1 : Examine the overall picture.
# Step 2 : Capture the numeric and categorical variables.
# Step 3 : Analyze the numerical and categorical variables.
# Step 4 : Perform target variable analysis. (The average
# of the target variable according to the categorical variables,
# the average of the numerical variables according to the target variable)
# Step 5 : Perform outlier observation analysis.
# Step 6 : Perform a missing observation analysis.
# Step 7 : Perform correlation analysis.

# Task 2 : Feature Engineering

# Step 1 :  Take the necessary actions for missing and contradictory observations.
# Step 2 : Create new variables.
# Step 3 : Perform the encoding operations.
# Step 4 : Standardize for numeric variables.
# Step 5 : Create the model.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import warnings
warnings.simplefilter("ignore")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def load_df():
    data = pd.read_csv(r"/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_6/datasets/Telco-Customer-Churn.csv")
    return data

df_ = load_df()
df = df_.copy()
df.head()

def check_df(dataframe, head=5, tail = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head ######################")
    print(dataframe.head(head))
    print("##################### Tail ######################")
    print(dataframe.tail(tail))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# ##################### Shape #####################
# (7043, 21)
# ##################### Types #####################
# customerID           object
# gender               object
# SeniorCitizen         int64
# Partner              object
# Dependents           object
# tenure                int64
# PhoneService         object
# MultipleLines        object
# InternetService      object
# OnlineSecurity       object
# OnlineBackup         object
# DeviceProtection     object
# TechSupport          object
# StreamingTV          object
# StreamingMovies      object
# Contract             object
# PaperlessBilling     object
# PaymentMethod        object
# MonthlyCharges      float64
# TotalCharges         object
# Churn                object
# dtype: object
# ##################### Head ######################
#    customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
# 0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check        29.85000        29.85    No
# 1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check        56.95000       1889.5    No
# 2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check        53.85000       108.15   Yes
# 3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)        42.30000      1840.75    No
# 4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check        70.70000       151.65   Yes
# ##################### Tail ######################
#       customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
# 7038  6840-RESVB    Male              0     Yes        Yes      24          Yes               Yes             DSL            Yes           No              Yes         Yes         Yes             Yes        One year              Yes               Mailed check        84.80000       1990.5    No
# 7039  2234-XADUH  Female              0     Yes        Yes      72          Yes               Yes     Fiber optic             No          Yes              Yes          No         Yes             Yes        One year              Yes    Credit card (automatic)       103.20000       7362.9    No
# 7040  4801-JZAZL  Female              0     Yes        Yes      11           No  No phone service             DSL            Yes           No               No          No          No              No  Month-to-month              Yes           Electronic check        29.60000       346.45    No
# 7041  8361-LTMKD    Male              1     Yes         No       4          Yes               Yes     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes               Mailed check        74.40000        306.6   Yes
# 7042  3186-AJIEK    Male              0      No         No      66          Yes                No     Fiber optic            Yes           No              Yes         Yes         Yes             Yes        Two year              Yes  Bank transfer (automatic)       105.65000       6844.5    No
# ##################### NA ########################
# customerID          0
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        0
# Churn               0
# dtype: int64
# ##################### Quantiles #####################
#                 0.00000  0.05000  0.50000   0.95000   0.99000   1.00000
# SeniorCitizen   0.00000  0.00000  0.00000   1.00000   1.00000   1.00000
# tenure          0.00000  1.00000 29.00000  72.00000  72.00000  72.00000
# MonthlyCharges 18.25000 19.65000 70.35000 107.40000 114.72900 118.75000

df.info()
# #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   customerID        7043 non-null   object
#  1   gender            7043 non-null   object
#  2   SeniorCitizen     7043 non-null   int64
#  3   Partner           7043 non-null   object
#  4   Dependents        7043 non-null   object
#  5   tenure            7043 non-null   int64
#  6   PhoneService      7043 non-null   object
#  7   MultipleLines     7043 non-null   object
#  8   InternetService   7043 non-null   object
#  9   OnlineSecurity    7043 non-null   object
#  10  OnlineBackup      7043 non-null   object
#  11  DeviceProtection  7043 non-null   object
#  12  TechSupport       7043 non-null   object
#  13  StreamingTV       7043 non-null   object
#  14  StreamingMovies   7043 non-null   object
#  15  Contract          7043 non-null   object
#  16  PaperlessBilling  7043 non-null   object
#  17  PaymentMethod     7043 non-null   object
#  18  MonthlyCharges    7043 non-null   float64
#  19  TotalCharges      7043 non-null   object
#  20  Churn             7043 non-null   object
# dtypes: float64(1), int64(2), object(18)

df.Contract.value_counts()
# Month-to-month    3875
# Two year          1695
# One year          1473

df.Churn.value_counts()
# No     5174
# Yes    1869

df.tenure.describe()
# count   7043.00000
# mean      32.37115
# std       24.55948
# min        0.00000
# 25%        9.00000
# 50%       29.00000
# 75%       55.00000
# max       72.00000

# show the distribution of tenure.
plt.hist(data = df, x = 'tenure');
plt.show()

# This is not a normal distribution, and with two peaks,
# which means there are likely two different kinds of groups of people,
# and either of them love particular services.

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.NAN)
df["TotalCharges"] = df["TotalCharges"].astype(float)

for col in df.columns:
    print(col + " : " + str((df[f"{col}"] == 0).sum()))
# customerID : 0
# gender : 0
# SeniorCitizen : 5901
# Partner : 0
# Dependents : 0
# tenure : 11
# PhoneService : 0
# MultipleLines : 0
# InternetService : 0
# OnlineSecurity : 0
# OnlineBackup : 0
# DeviceProtection : 0
# TechSupport : 0
# StreamingTV : 0
# StreamingMovies : 0
# Contract : 0
# PaperlessBilling : 0
# PaymentMethod : 0
# MonthlyCharges : 0
# TotalCharges : 0
# Churn : 0

df = df[df["tenure"] != 0]

df.isnull().any()
# customerID          False
# gender              False
# SeniorCitizen       False
# Partner             False
# Dependents          False
# tenure              False
# PhoneService        False
# MultipleLines       False
# InternetService     False
# OnlineSecurity      False
# OnlineBackup        False
# DeviceProtection    False
# TechSupport         False
# StreamingTV         False
# StreamingMovies     False
# Contract            False
# PaperlessBilling    False
# PaymentMethod       False
# MonthlyCharges      False
# TotalCharges        False
# Churn               False

df.isnull().sum()
# customerID          0
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        0
# Churn               0

df.dropna(inplace=True)

del df["customerID"]
del df["SeniorCitizen"]

df.TotalCharges.describe()
# count   7032.00000
# mean    2283.30044
# std     2266.77136
# min       18.80000
# 25%      401.45000
# 50%     1397.47500
# 75%     3794.73750
# max     8684.80000

plt.hist(data = df, x = 'TotalCharges');
plt.show()

churn_df = df.query('Churn=="Yes"')

churn_df.TotalCharges.describe()
# count   1869.00000
# mean    1531.79609
# std     1890.82299
# min       18.85000
# 25%      134.50000
# 50%      703.55000
# 75%     2331.30000
# max     8684.80000

plt.hist(data = churn_df, x = 'TotalCharges');
plt.show()

# The found that around 20% of the data are extremely high,
# so I decided to divide them to see each distribution of data.

churn_df.TotalCharges.quantile(0.8)
# 2840.4100000000003

# Divide the data by the 80th percentile of the data,
# and show the distribution of its TotalCharges under 80th percentile
TotalCharges_under80 = churn_df.query('TotalCharges<=2827.59')
TotalCharges_above80 = churn_df.query('TotalCharges>2827.59')
TotalCharges_under80.TotalCharges.describe()
# count   1493.00000
# mean     708.41530
# std      763.39151
# min       18.85000
# 25%       85.00000
# 50%      370.65000
# 75%     1127.20000
# max     2816.65000

TotalCharges_above80.TotalCharges.describe()
# count    376.00000
# mean    4801.23098
# std     1440.07259
# min     2838.70000
# 25%     3510.51250
# 50%     4544.22500
# 75%     5887.33750
# max     8684.80000

TotalCharges_under80.tenure.describe()
# count   1493.00000
# mean       9.90355
# std       10.71405
# min        1.00000
# 25%        1.00000
# 50%        6.00000
# 75%       15.00000
# max       61.00000

TotalCharges_above80.tenure.describe()
# count   376.00000
# mean     50.04521
# std      12.36406
# min      27.00000
# 25%      40.00000
# 50%      49.00000
# 75%      60.00000
# max      72.00000

# Visualize both
plt.figure(figsize = [15, 5])
plt.show()

# left plot: LTV by above and under 80th percentile of data who unsubscribed
plt.subplot(1, 2, 1)
plt.bar([1, 2], [713, 4801])
plt.show()

# # right plot: Tenure by above and under 80th percentile of data who unsubscribed
plt.subplot(1, 2, 2)
plt.bar([1, 2], [9, 50])
plt.show()

# Note: The average LTV of 80% of those who unsubscribed is 750 dollars,
# and its tenures is near 10 months. On the other hand, the average LTV of top 20% of those who unsubscribed is 4750 dollars,
# and its tenures is near 50 months. And the ratio by the sum of total LTV by each groups is 750*4 : 4750 = 1 : 1.6,
# which suggests we should focus on serving those 20% customers with high LTV, which brought 60%(1.6/2.6) of our revennue.

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

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
    return cat_cols, num_cols, cat_but_car

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Observations: 7032
# Variables: 19
# cat_cols: 16
# num_cols: 3
# cat_but_car: 0
# num_but_cat: 0


check_outlier(df, col)
# False

replace_with_thresholds(df, col)

num_cols
# ['tenure', 'MonthlyCharges', 'TotalCharges']

cat_cols
# ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
# 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
# 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
# 'PaymentMethod', 'Churn', 'SeniorCitizen']

cat_but_car

df["Churn"].replace(["Yes"], "1", inplace=True)
df["Churn"].replace(["No"], "0", inplace=True)
df["Churn"] = df["Churn"].astype(int)

def num_summary(dataframe, numerical_col, plot=True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.show()

for col in num_cols:
    num_summary(df, col)
# count   7032.00000
# mean      32.42179
# std       24.54526
# min        1.00000
# 5%         1.00000
# 10%        2.00000
# 20%        6.00000
# 30%       12.00000
# 40%       20.00000
# 50%       29.00000
# 60%       40.00000
# 70%       50.00000
# 80%       60.80000
# 90%       69.00000
# 95%       72.00000
# 99%       72.00000
# max       72.00000
# Name: tenure, dtype: float64
# count   7032.00000
# mean      64.79821
# std       30.08597
# min       18.25000
# 5%        19.65000
# 10%       20.05000
# 20%       25.05000
# 30%       45.90000
# 40%       58.92000
# 50%       70.35000
# 60%       79.15000
# 70%       85.53500
# 80%       94.30000
# 90%      102.64500
# 95%      107.42250
# 99%      114.73450
# max      118.75000
# Name: MonthlyCharges, dtype: float64
# count   7032.00000
# mean    2283.30044
# std     2266.77136
# min       18.80000
# 5%        49.60500
# 10%       84.60000
# 20%      267.07000
# 30%      551.99500
# 40%      944.17000
# 50%     1397.47500
# 60%     2048.95000
# 70%     3141.13000
# 80%     4475.41000
# 90%     5976.64000
# 95%     6923.59000
# 99%     8039.88300
# max     8684.80000
# Name: TotalCharges, dtype: float64

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)
#         gender    Ratio
# Male      3549 50.46928
# Female    3483 49.53072
# ##########################################
#      Partner    Ratio
# No      3639 51.74915
# Yes     3393 48.25085
# ##########################################
#      Dependents    Ratio
# No         4933 70.15074
# Yes        2099 29.84926
# ##########################################
#      PhoneService    Ratio
# Yes          6352 90.32992
# No            680  9.67008
# ##########################################
#                   MultipleLines    Ratio
# No                         3385 48.13709
# Yes                        2967 42.19283
# No phone service            680  9.67008
# ##########################################
#              InternetService    Ratio
# Fiber optic             3096 44.02730
# DSL                     2416 34.35722
# No                      1520 21.61547
# ##########################################
#                      OnlineSecurity    Ratio
# No                             3497 49.72981
# Yes                            2015 28.65472
# No internet service            1520 21.61547
# ##########################################
#                      OnlineBackup    Ratio
# No                           3087 43.89932
# Yes                          2425 34.48521
# No internet service          1520 21.61547
# ##########################################
#                      DeviceProtection    Ratio
# No                               3094 43.99886
# Yes                              2418 34.38567
# No internet service              1520 21.61547
# ##########################################
#                      TechSupport    Ratio
# No                          3472 49.37429
# Yes                         2040 29.01024
# No internet service         1520 21.61547
# ##########################################
#                      StreamingTV    Ratio
# No                          2809 39.94596
# Yes                         2703 38.43857
# No internet service         1520 21.61547
# ##########################################
#                      StreamingMovies    Ratio
# No                              2781 39.54778
# Yes                             2731 38.83675
# No internet service             1520 21.61547
# ##########################################
#                 Contract    Ratio
# Month-to-month      3875 55.10523
# Two year            1685 23.96189
# One year            1472 20.93288
# ##########################################
#      PaperlessBilling    Ratio
# Yes              4168 59.27190
# No               2864 40.72810
# ##########################################
#                            PaymentMethod    Ratio
# Electronic check                    2365 33.63197
# Mailed check                        1604 22.81001
# Bank transfer (automatic)           1542 21.92833
# Credit card (automatic)             1521 21.62969
# ##########################################
#    Churn    Ratio
# 0   5163 73.42150
# 1   1869 26.57850
# ##########################################
#    SeniorCitizen    Ratio
# 0           5890 83.75995
# 1           1142 16.24005
# ##########################################

def target_analyser(dataframe, target, num_cols, cat_cols):
    for col in dataframe.columns:
        if col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        if col in num_cols:
            print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(target)[col].mean()}), end="\n\n\n")

target_analyser(df, "Churn", num_cols, cat_cols)
# gender : 2
#         COUNT   RATIO  TARGET_MEAN
# Female   3483 0.49531      0.26960
# Male     3549 0.50469      0.26205
# SeniorCitizen : 2
#    COUNT   RATIO  TARGET_MEAN
# 0   5890 0.83760      0.23650
# 1   1142 0.16240      0.41681
# Partner : 2
#      COUNT   RATIO  TARGET_MEAN
# No    3639 0.51749      0.32976
# Yes   3393 0.48251      0.19717
# Dependents : 2
#      COUNT   RATIO  TARGET_MEAN
# No    4933 0.70151      0.31279
# Yes   2099 0.29849      0.15531
#        TARGET_MEAN
# Churn
# 0         37.65001
# 1         17.97913
# PhoneService : 2
#      COUNT   RATIO  TARGET_MEAN
# No     680 0.09670      0.25000
# Yes   6352 0.90330      0.26747
# MultipleLines : 3
#                   COUNT   RATIO  TARGET_MEAN
# No                 3385 0.48137      0.25081
# No phone service    680 0.09670      0.25000
# Yes                2967 0.42193      0.28648
# InternetService : 3
#              COUNT   RATIO  TARGET_MEAN
# DSL           2416 0.34357      0.18998
# Fiber optic   3096 0.44027      0.41893
# No            1520 0.21615      0.07434
# OnlineSecurity : 3
#                      COUNT   RATIO  TARGET_MEAN
# No                    3497 0.49730      0.41779
# No internet service   1520 0.21615      0.07434
# Yes                   2015 0.28655      0.14640
# OnlineBackup : 3
#                      COUNT   RATIO  TARGET_MEAN
# No                    3087 0.43899      0.39942
# No internet service   1520 0.21615      0.07434
# Yes                   2425 0.34485      0.21567
# DeviceProtection : 3
#                      COUNT   RATIO  TARGET_MEAN
# No                    3094 0.43999      0.39140
# No internet service   1520 0.21615      0.07434
# Yes                   2418 0.34386      0.22539
# TechSupport : 3
#                      COUNT   RATIO  TARGET_MEAN
# No                    3472 0.49374      0.41647
# No internet service   1520 0.21615      0.07434
# Yes                   2040 0.29010      0.15196
# StreamingTV : 3
#                      COUNT   RATIO  TARGET_MEAN
# No                    2809 0.39946      0.33535
# No internet service   1520 0.21615      0.07434
# Yes                   2703 0.38439      0.30115
# StreamingMovies : 3
#                      COUNT   RATIO  TARGET_MEAN
# No                    2781 0.39548      0.33729
# No internet service   1520 0.21615      0.07434
# Yes                   2731 0.38837      0.29952
# Contract : 3
#                 COUNT   RATIO  TARGET_MEAN
# Month-to-month   3875 0.55105      0.42710
# One year         1472 0.20933      0.11277
# Two year         1685 0.23962      0.02849
# PaperlessBilling : 2
#      COUNT   RATIO  TARGET_MEAN
# No    2864 0.40728      0.16376
# Yes   4168 0.59272      0.33589
# PaymentMethod : 4
#                            COUNT   RATIO  TARGET_MEAN
# Bank transfer (automatic)   1542 0.21928      0.16732
# Credit card (automatic)     1521 0.21630      0.15253
# Electronic check            2365 0.33632      0.45285
# Mailed check                1604 0.22810      0.19202
#        TARGET_MEAN
# Churn
# 0         61.30741
# 1         74.44133
#        TARGET_MEAN
# Churn
# 0       2555.34414
# 1       1531.79609
# Churn : 2
#    COUNT   RATIO  TARGET_MEAN
# 0   5163 0.73422      0.00000
# 1   1869 0.26578      1.00000

outlier_thresholds(df, col)
# (-1.5, 2.5)

check_outlier(df, col)
#False

missing_values_table(df)

cor = df.corr(method="pearson")
cor
#                  tenure  MonthlyCharges  TotalCharges    Churn
# tenure          1.00000         0.24686       0.82588 -0.35405
# MonthlyCharges  0.24686         1.00000       0.65106  0.19286
# TotalCharges    0.82588         0.65106       1.00000 -0.19948
# Churn          -0.35405         0.19286      -0.19948  1.00000

sns.heatmap(cor)
plt.show()

def pairplot(dataset, target_column):
    sns.set(style="ticks")
    sns.pairplot(dataset, hue=target_column)
    plt.show()

pairplot(df, "Churn")

df.loc[df["tenure"] <= 12 , "tenure_cat"] = "one year customer"
df.loc[((df["tenure"] <= 24) & (df["tenure"] > 12)), "tenure_cat"] = "two year customer"
df.loc[((df["tenure"] <= 36) & (df["tenure"] > 24)), "tenure_cat"] = "three year customer"
df.loc[((df["tenure"] <= 60) & (df["tenure"] > 36)), "tenure_cat"] = "five year customer"
df.loc[df["tenure"] > 60, "tenure_cat"] = "over five year customer"

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols = [col for col in cat_cols if col not in cat_but_car]
ohe_cols = [col for col in cat_cols if col not in ["Churn"]]

one_hot_encoder(df, ohe_cols)
#       tenure  MonthlyCharges  TotalCharges  Churn  gender_Male  Partner_Yes  Dependents_Yes  PhoneService_Yes  MultipleLines_No phone service  MultipleLines_Yes  InternetService_Fiber optic  InternetService_No  OnlineSecurity_No internet service  OnlineSecurity_Yes  OnlineBackup_No internet service  OnlineBackup_Yes  DeviceProtection_No internet service  DeviceProtection_Yes  TechSupport_No internet service  TechSupport_Yes  StreamingTV_No internet service  StreamingTV_Yes  StreamingMovies_No internet service  StreamingMovies_Yes  Contract_One year  Contract_Two year  PaperlessBilling_Yes  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  tenure_cat_one year customer  tenure_cat_over five year customer  tenure_cat_three year customer  tenure_cat_two year customer
# 0          1        29.85000      29.85000      0            0            1               0                 0                               1                  0                            0                   0                                   0                   0                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0
# 1         34        56.95000    1889.50000      0            1            0               0                 1                               0                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                0                                0                0                                    0                    0                  1                  0                     0                                      0                               0                           1                             0                                   0                               1                             0
# 2          2        53.85000     108.15000      1            1            0               0                 1                               0                  0                            0                   0                                   0                   1                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               0                           1                             1                                   0                               0                             0
# 3         45        42.30000    1840.75000      0            1            0               0                 0                               1                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                0                                    0                    0                  1                  0                     0                                      0                               0                           0                             0                                   0                               0                             0
# 4          2        70.70000     151.65000      1            0            0               0                 1                               0                  0                            1                   0                                   0                   0                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0
#       ...             ...           ...    ...          ...          ...             ...               ...                             ...                ...                          ...                 ...                                 ...                 ...                               ...               ...                                   ...                   ...                              ...              ...                              ...              ...                                  ...                  ...                ...                ...                   ...                                    ...                             ...                         ...                           ...                                 ...                             ...                           ...
# 7038      24        84.80000    1990.50000      0            1            1               1                 1                               0                  1                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                1                                    0                    1                  1                  0                     1                                      0                               0                           1                             0                                   0                               0                             1
# 7039      72       103.20000    7362.90000      0            0            1               1                 1                               0                  1                            1                   0                                   0                   0                                 0                 1                                     0                     1                                0                0                                0                1                                    0                    1                  1                  0                     1                                      1                               0                           0                             0                                   1                               0                             0
# 7040      11        29.60000     346.45000      0            0            1               1                 0                               1                  0                            0                   0                                   0                   1                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0
# 7041       4        74.40000     306.60000      1            1            1               0                 1                               0                  1                            1                   0                                   0                   0                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               0                           1                             1                                   0                               0                             0
# 7042      66       105.65000    6844.50000      0            1            0               0                 1                               0                  0                            1                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                1                                    0                    1                  0                  1                     1                                      0                               0                           0                             0                                   1                               0                             0
# [7032 rows x 34 columns]

df = one_hot_encoder(df, ohe_cols)

labelencoder = LabelEncoder()
df["Churn"] = labelencoder.fit_transform(df["Churn"])
df.head()
#    tenure  MonthlyCharges  TotalCharges  Churn  gender_Male  Partner_Yes  Dependents_Yes  PhoneService_Yes  MultipleLines_No phone service  MultipleLines_Yes  InternetService_Fiber optic  InternetService_No  OnlineSecurity_No internet service  OnlineSecurity_Yes  OnlineBackup_No internet service  OnlineBackup_Yes  DeviceProtection_No internet service  DeviceProtection_Yes  TechSupport_No internet service  TechSupport_Yes  StreamingTV_No internet service  StreamingTV_Yes  StreamingMovies_No internet service  StreamingMovies_Yes  Contract_One year  Contract_Two year  PaperlessBilling_Yes  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  tenure_cat_one year customer  tenure_cat_over five year customer  tenure_cat_three year customer  tenure_cat_two year customer
# 0       1        29.85000      29.85000      0            0            1               0                 0                               1                  0                            0                   0                                   0                   0                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0
# 1      34        56.95000    1889.50000      0            1            0               0                 1                               0                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                0                                0                0                                    0                    0                  1                  0                     0                                      0                               0                           1                             0                                   0                               1                             0
# 2       2        53.85000     108.15000      1            1            0               0                 1                               0                  0                            0                   0                                   0                   1                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               0                           1                             1                                   0                               0                             0
# 3      45        42.30000    1840.75000      0            1            0               0                 0                               1                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                0                                    0                    0                  1                  0                     0                                      0                               0                           0                             0                                   0                               0                             0
# 4       2        70.70000     151.65000      1            0            0               0                 1                               0                  0                            1                   0                                   0                   0                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0

num_cols

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()
#     tenure  MonthlyCharges  TotalCharges  Churn  gender_Male  Partner_Yes  Dependents_Yes  PhoneService_Yes  MultipleLines_No phone service  MultipleLines_Yes  InternetService_Fiber optic  InternetService_No  OnlineSecurity_No internet service  OnlineSecurity_Yes  OnlineBackup_No internet service  OnlineBackup_Yes  DeviceProtection_No internet service  DeviceProtection_Yes  TechSupport_No internet service  TechSupport_Yes  StreamingTV_No internet service  StreamingTV_Yes  StreamingMovies_No internet service  StreamingMovies_Yes  Contract_One year  Contract_Two year  PaperlessBilling_Yes  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  tenure_cat_one year customer  tenure_cat_over five year customer  tenure_cat_three year customer  tenure_cat_two year customer
# 0 -0.60870        -0.74620      -0.40304      0            0            1               0                 0                               1                  0                            0                   0                                   0                   0                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0
# 1  0.10870        -0.24689       0.14500      0            1            0               0                 1                               0                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                0                                0                0                                    0                    0                  1                  0                     0                                      0                               0                           1                             0                                   0                               1                             0
# 2 -0.58696        -0.30401      -0.37996      1            1            0               0                 1                               0                  0                            0                   0                                   0                   1                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               0                           1                             1                                   0                               0                             0
# 3  0.34783        -0.51681       0.13063      0            1            0               0                 0                               1                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                0                                    0                    0                  1                  0                     0                                      0                               0                           0                             0                                   0                               0                             0
# 4 -0.58696         0.00645      -0.36714      1            0            0               0                 1                               0                  0                            1                   0                                   0                   0                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                     1                                      0                               1                           0                             1                                   0                               0                             0

y = df["Churn"]
X = df.drop(["Churn", "tenure"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=47)



rf_model = RandomForestClassifier(random_state=5).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# 0.7666666666666667

print("Accuracy Score: " + f'{accuracy_score(y_pred, y_test):.2f}')
# Accuracy Score: 0.77

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

plot_importance(rf_model, X_train)






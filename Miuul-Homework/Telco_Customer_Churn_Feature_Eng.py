#######################
#Telco Churn Feature Engineering
#######################

####################
# BUSINESS PROBLEM
####################
# It is desired to develop a machine learning model that can predict customers who will leave the company.
# You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

#######################
# Dataset Story
#######################
# Telco customer churn data includes fictitious telco information that provided home phone and Internet service to 7,043 customers in California in the third quarter.
# Shows which customers left, stayed, or signed up for their service.
#21 Variable 7043 Observations 977.5 KB

# CustomerId : Customer ID
# Gender
# SeniorCitizen : Whether the customer is elderly (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents Whether the customer has dependents (Yes, No
# Tenure: The number of months the customer stays with the company
# PhoneService: Whether the customer has phone service (Yes, No)
# MultipleLines: Whether the customer has more than one line (Yes, No, No phone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, no Internet service)
# OnlineBackup: Whether the customer has an online backup (Yes, No, no Internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, no Internet service)
# TechSupport: Whether the customer receives technical support (Yes, No, no Internet service)
# StreamingTV : Whether the customer has TV broadcasting (Yes, No, no Internet service)
# StreamingMovies : Whether the customer is streaming movies (Yes, No, No Internet service)
# Contract : Customer's contract period (Month to month, One year, Two years)
# PaperlessBilling: Whether the customer has a paperless bill (Yes, No)
# PaymentMethod :Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: The amount collected from the customer monthly
# TotalCharges: Total amount charged from the customer
# Churn: Whether the customer uses it (Yes or No)

#############################
# Project Tasks
#############################
# Required library and functions

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from catboost import CatBoostClassifier
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 300)

df= pd.read_csv("Feature_Engineering/datasets/Telco-Customer-Churn.csv")
df.head()

df.isnull().sum()
df.shape
df.info()

#TotalCharges must be a numeric variable
df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors="coerce")

df["Churn"]= df["Churn"].apply(lambda x: 1 if x== "Yes" else 0)

####################
# Task 1: Exploratory Data Analysis
####################
# Step 1: Examine the general picture.

def check_df(dataframe, head=5):
     print("################################## Shape #################")
     print(dataframe.shape)
     print("################################## Types #################")
     print(dataframe.dtypes)
     print("################################## Head ##################")
     print(dataframe.head())
     print("################################## Tail ################")
     print(dataframe.tail())
     print("##################################NA ##################")
     print(dataframe.isnull().sum())
     print("################################## Quantiles #################")
     print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99,1]).T)

check_df(df)

df.head()
df.info()

# Step 2: Capture numerical and categorical variables.
def grab_col_names(dataframe, cat_th=10, car_th=20):
# We determined the threshold values ourselves by checking the acceptance above these ranges
# Important information###
"""

It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
Note: Categorical variables with numerical appearance are also included.

parameters
------
    dataframe: dataframe
            Dataframe from which variable names are to be taken
    cat_th: int, optional
            Class threshold value for variables that are numeric but categorical
    car_th: int, optinal
            class threshold for categorical but cardinal variables

returns
------
    cat_cols: list
            Categorical variable list
    num_cols: list
            Numerical variable list
    cat_but_car: list
            List of cardinal variables with categorical view

examples
------
    import seaborn as sns
    df = sns.load_dataset("iris")
    print(grab_col_names(df))


Notes
------
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is inside cat_cols.
    The sum of the 3 lists that return equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


# Step 3: Analyze numerical and categorical variables.
# Analysis of Categorical Variables
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################################"
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.show()

    for col in cat_cols:
        cat_summary(df, col)

        # Analysis of Numerical Variables


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)


# Step 4: Perform target variable analysis.
# #(Average of target variable according to categorical variables, average of numerical variables according to target variable)
# Analysis of Numerical Variables According to Target // How many months has tenure:musterı been with us?
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)


# Analysis of Categorical Variables According to Target
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Step 5: Perform an outlier observation analysis.


# Step 6: Perform missing observation analysis.
df.isnull().sum()

# Step 7: Perform correlation analysis.
# Correlation
df.corr()
# Correlation matrix
f, ax = plt.subplots(figsize=[10, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers appears to be highly correlated with monthly charges and tenure
df.corrwith(df["Churn"]).sort_values(ascending=False)

####################
# Task 2: Feature Engineering
####################
# Step 1: Take the necessary action for missing and contradictory observations.

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns=missing_values_table(df,na_name=True)

# Total charge can be filled with monthly payment amounts (may be better) or 11 variables can be dropped.

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.isnull().sum()

# BASE MODEL SETUP

dff = df.copy()
cat_cols=[col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe
dff= one_hot_encoder(dff, cat_cols, drop_first=True)

df.head()

y= dff["Churn"]
X= dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred= catboost_model.predict(X_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 3)}")
print(f"F1:{round(f1_score(y_pred, y_test), 2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test), 2)}")



#############################
# OUTLIER VALUE ANALYSIS
#############################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

def replace_with_thresholds(dataframe, variable):
     low_limit, up_limit = outlier_thresholds(dataframe, variable)
     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Outlier analysis and Suppression Process
for col in num_cols:
     print(col, check_outlier(df,col))
     if check_outlier(df, col):
         replace_with_thresholds(df, col)


# Step 2: Create new variables.

##############
# FEATURE EXTRACTION
##############

# Creating annual categorical variable from Tenure variable
df.loc[(df["tenure"]>=0) & (df["tenure"] <=12), "NEW_TENURE_YEAR"]= "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"] <=24), "NEW_TENURE_YEAR"]= "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"] <=36), "NEW_TENURE_YEAR"]= "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"] <=48), "NEW_TENURE_YEAR"]= "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"] <=60), "NEW_TENURE_YEAR"]= "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"] <=72), "NEW_TENURE_YEAR"]= "5-6 Year"

# Specify customers with 1 or 2 year contracts as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x:1 if x in["One year","Two year"] else 0)

# People who do not receive any support, backup or protection
df["NEW_noProt"]= df.apply(lambda x:1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport" ] != "Yes") else 0, axis=1)

# Customers who have monthly contracts and are young
df["NEW_Young_Not_Engaged"]= df.apply(lambda x:1 if (x["NEW_Engaged"]== 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Total number of services received by the person
df["New_TotalServices"] = (df[["PhoneService", "InternetService", "OnlineSecurity",
                                "OnlineBackup","DeviceProtection","TechSupport",
                                "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

# Total number of services received by the person
df["NEW_FLAG_ANY_STREAMING"]= df.apply(lambda x:1 if(x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Does the person make automatic payments?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x:1 if x in ["Bank transfer(automatic)","Credit card (automatic)"] else 0)

# Average Monthly payment
df["New_AVG_Charges"]= df["TotalCharges"] / (df["tenure"]+1)

# Increase of the current price compared to the average price
df["NEW_Increase"]= df["New_AVG_Charges"] / df["MonthlyCharges"]

# Fee per service
df["New_AVG_Service_Fee"]= df["MonthlyCharges"] / (df["New_TotalServices"] + 1)

df.head()
df.shape

# Step 3: Perform the encoding operations.
cat_cols, num_cols, cat_but_car= grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
     labelencoder = LabelEncoder()
     dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
     return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes =="O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
     label_encoder(df, col)

# One Hot Encoding Process
# update process of cat_cols list
cat_cols= [col for col in cat_cols if col not in binary_cols and col not in["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape

# Step 4: Standardize numerical variables.

# Step 5: Create the model.

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 3)}")
print(f"F1:{round(f1_score(y_pred, y_test), 2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test), 2)}")




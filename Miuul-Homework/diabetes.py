######
# Diabetes Future Engineering
######

####################
# BUSINESS PROBLEM
####################
# It is desired to develop a machine learning model that can predict whether people have diabetes or not when their characteristics are specified.
# You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

#######################
# Dataset Story
#######################
# The dataset is part of a larger dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the US. in the USA
# Data used for diabetes research on Pima Indian women aged 21 and over living in the city of Phoenix, the 5th largest city in the State of Arizona.
# The target variable is specified as "outcome"; 1 indicates a positive diabetes test result, and 0 indicates a negative diabetes test result.

#9 Variable 768 Observations 24 KB
# Pregnancies : Number of pregnancies
# Glucose Oral: 2-hour plasma glucose concentration in glucose tolerance test
# Blood Pressure: Blood Pressure (diastolic blood pressure) (mm Hg)
# SkinThickness : Skin Thickness
# Insulin: 2-hour serum insulin (mu U/ml)
# DiabetesPedigreeFunction : A function that calculates the probability of having diabetes based on people in the ancestry
# BMI : Body mass index
# Age : Age (years)
# Outcome: Having the disease (1) or not (0)
#############################
# Project Tasks
#############################
# Required library and functions

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 300)

df= pd.read_csv("Feature_Engineering/datasets/diabetes.csv")
df.head()

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
     #Important information###
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

# Categorical Variable Analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################################"
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.show()

    cat_summary(df, "Outcome")

    # In case of more than one
    for col in cat_cols:
        cat_summary(df, col)

        # Numerical Variable Analysis


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


# Step 4: Perform target variable analysis. (Average of target variable according to categorical variables, average of numerical variables according to target variable)
# Analysis of Numerical Variables According to Target
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Step 5: Perform an outlier observation analysis.

# Step 6: Perform missing observation analysis.
df.isnull().sum()
# Step 7: Perform correlation analysis.

# CORRELATION
# In probability theory and statistics, it indicates the direction and strength of the linear relationship between two random variables.
# Correlation is between 1 and -1, the ratio towards 1 is high, the correlation strength is high.
# Closer to -1, the higher the relationship strength is inversely proportional.
# As it approaches 0, it means that there is no relationship strength, there are variables that are not similar to each other.
df.corr()

# Correlation matrix
f, ax = plt.subplots(figsize=[10, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Model Setup
y = df["Outcome"]
x = df.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 3)}")
print(f"F1:{round(f1_score(y_pred, y_test), 2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test), 2)}")


# Accuracy:0.77
# Recall:0.706 = how successfully the positive class was predicted
# F1:0.64 = success of positive predicted values
# Auc:0.75

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_model, x)

####################
# Task 1: Feature Engineering
####################

# Step 1: Take the necessary action for missing and outlier values.
# -There are no missing observations in the data set, but Glucose, Insulin etc.
# - Observation units containing 0 values in variables may represent missing values. \
#  -For example; A person's glucose or insulin value will not be 0.
# -Taking this situation into consideration, you can assign zero values as NaN in the relevant values and then apply the operations to the missing values.
#############################
# MISSING VALUE ANALYSIS
#############################
df.isnull().sum()
df.describe()

zero_columns=[col for col in df.columns if(df[col].min() == 0 and col not in ["Pregnancies","Outcome"])]
zero_columns

# We go to each variable whose observation units are 0 and replace the observation values containing 0 with Nan.
for col in zero_columns:
     df[col]= np.where(df[col] == 0, np.nan, df[col])

# Missing Observation Analysis
df.isnull().sum()

# Number of missing values and their ratio to all data
def missing_values_table(dataframe, na_name=False):
     na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

     n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
     ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
     missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
     print(missing_df, end="\n")
     if na_name:
         return na_columns

na_columns=missing_values_table(df,na_name=True)

# Examination of Missing Values in Relation to the Dependent Variable

def missing_vs_target(dataframe, target, na_columns):
     temp_df = dataframe.copy()

     for col in na_columns:
         temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

     na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
     for col in na_flags:
         print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                             "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)

# Filling in Missing Values:
for col in zero_columns:
     df.loc[df[col].isnull(), col]= df[col].median()

df.isnull().sum()

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
for col in df.columns:
     print(col, check_outlier(df,col))
     if check_outlier(df, col):
         replace_with_thresholds(df, col)

# see if there are any outliers with check outlier
for col in df.columns:
     print(col, check_outlier(df,col))

# Step 2: Create new variables.
# Separating the age variable into new categories and creating a new age variable
df.loc[(df['Age'] >= 21 ) & (df['Age'] < 50), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 50 ), 'NEW_AGE_CAT'] = 'senior'

df.head()

# BMI below 18.5 is underweight, between 18.5 and 24.9 is normal, between 24.9 and 29.9 is overweight and over 30 is obese
df["NEW_BMI"]= pd.cut(x=df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "healthy", "Overweight", " Obese"])

# Convert glucose value to categorical variable
df["NEW_GLUCOSE"]= pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Creating a categorical variable by considering age and body mass indices together. 3 breakdowns were achieved
df.loc[(df["BMI"] < 18.5) & ((df['Age'] >= 21) & (df['Age'] < 50)), "NEW_AGE_BMI_NOM"] = 'Underweightmature'
df.loc[(df["BMI"] < 18.5) & (df['Age'] >= 50), "NEW_AGE_BMI_NOM"] = 'Underweightsenior'
df.loc[((df["BMI"] >= 18.5) & (df['BMI'] < 25)) & ((df['Age'] >= 21) & (df["Age"]< 50)), "NEW_AGE_BMI_NOM"] = 'healthymature'
df.loc[((df["BMI"] >= 18.5) & (df['BMI'] < 25)) & (df['Age'] >= 50), "NEW_AGE_BMI_NOM"] = 'healthysenior'
df.loc[((df["BMI"] >= 25) & (df['BMI'] < 30)) & ((df['Age'] >= 21) & (df["Age"]< 50)), "NEW_AGE_BMI_NOM"] = 'overweightmature'
df.loc[((df["BMI"] >= 25) & (df['BMI'] < 30)) & (df['Age'] >= 50), "NEW_AGE_BMI_NOM"] = 'overweightsenior'
df.loc[(df["BMI"] > 18.5) & ((df['Age'] >= 21) & (df['Age'] < 50)), "NEW_AGE_BMI_NOM"] = 'obesemature'
df.loc[(df["BMI"] > 18.5) & (df['Age'] >= 50), "NEW_AGE_BMI_NOM"] = 'obesesenior'

# Creating a variable by considering Age and Glucose values together
df.loc[(df["Glucose"] < 70) & ((df['Age'] >= 21) & (df['Age'] < 50)), "NEW_AGE_GLUCOSE_NOM"] = 'lowmature'
df.loc[(df["Glucose"] < 70) & (df['Age'] >= 50), "NEW_AGE_GLUCOSE_NOM"] = 'lowsenior'
df.loc[((df["Glucose"] >= 70) & (df['Glucose'] < 100)) & ((df['Age'] >= 21) & (df['Age'] < 50)), "NEW_AGE_GLUCOSE_NOM"] = 'normalmature'
df.loc[((df["Glucose"] >= 70) & (df['Glucose'] < 100)) & (df['Age'] >= 50), "NEW_AGE_GLUCOSE_NOM"] = 'normalsenior'
df.loc[((df["Glucose"] >= 100) & (df['Glucose'] <= 125)) & ((df['Age'] >= 21) & (df['Age'] < 50)), "NEW_AGE_GLUCOSE_NOM"] = 'hiddenmature'
df.loc[((df["Glucose"] >= 100) & (df['Glucose'] <= 125)) & (df['Age'] >= 50), "NEW_AGE_GLUCOSE_NOM"] = 'hiddensenior'
df.loc[(df["Glucose"] > 125) & ((df['Age'] >=21 ) & (df['Age'] < 50)), "NEW_AGE_GLUCOSE_NOM"] = 'highmature'
df.loc[(df["Glucose"] > 125) & (df['Age'] >=50 ),"NEW_AGE_GLUCOSE_NOM"] = 'highsenior'

#Creating a categorical variable with Insulin Value
def set_insulin(dataframe, col_name="Insulin"):
     if 16 <= dataframe[col_name] <= 166:
         return "Normal"
     else:
         return "Abnormal"

df["NEW_INSULIN_SCORE"]=df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE+INSULIN"]=df["Glucose"]* df["Insulin"]

df.head()

# Attention to zero values!!
df["NEW_GLUCOSE+PREGNANCIES"]= df["Glucose"] * df["Pregnancies"]
# df["NEW_GLUCOSE+PREGNANCIES"]= df["Glucose"] * (1 + df["Pregnancies"])

#Enlargement of colons
df.columns=[col.upper() for col in df.columns]
df.head()

# Step 3: Perform the encoding operations.
##########
# ENCODING
##########

# Separation of variables according to their types
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
cat_cols= [col for col in cat_cols if col not in binary_cols and col not in["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe

df= one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# Standardization

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

# Step 4: Standardize numerical variables.

# Step 5: Create the model.

##########
# MODELLING
##########
y= df["OUTCOME"]
x= df.drop("OUTCOME", axis=1)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30, random_state=17)

rf_model=RandomForestClassifier(random_state=46).fit(x_train,y_train)
y_pred= rf_model.predict(x_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 3)}")
print(f"F1:{round(f1_score(y_pred, y_test), 2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test), 2)}")


#############
#FEAUTURE IMPORTANTS
###########

def plot_importance(model, features, num=len(x), save=False):
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


plot_importance(rf_model, x_train)

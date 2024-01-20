#######################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#######################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
     data = pd.read_csv("Feature_Engineering/datasets/application_train.csv")
     return data

df = load_application_train()
df.head()

def load():
     data = pd.read_csv("Feature_Engineering/datasets/titanic.csv")
     return data


df = load()
df.head()

#######################
#1. Outliers
#######################

#######################
# Capturing Outliers
#######################

####################
# Outliers with Chart Technique
####################

#Box plot is used for numerical variables

sns.boxplot(x=df["Age"])
plt.show()

####################
# How to Catch Outliers?
####################

#First quartile values
q1 = df["Age"].quantile(0.25)

#Second quarter values
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index


####################
# Are There Any Outliers?
####################
# If I receive many rows as a result of the operation, I will be informed whether there are any rows there.

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[(df["Age"] < low)].any(axis=None)

# 1. We set a threshold value.
#2. We reached the outliers.
#3. We quickly asked if there were any outliers.

####################
# Functionalize Transactions
####################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
     quartile1 = dataframe[col_name].quantile(q1)
     quartile3 = dataframe[col_name].quantile(q3)
     interquantile_range = quartile3 - quartile1
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Mouse")

low, up = outlier_thresholds(df, "Mouse")

df[(df["Mouse"] < low) | (df["Mouse"] > up)].head()


df[(df["Mouse"] < low) | (df["Mouse"] > up)].index

# Function to ask whether there are outliers or not

def check_outlier(dataframe, col_name):
     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
         return True
     else:
         return False

check_outlier(df, "Age")
check_outlier(df, "Mouse")

####################
#grab_col_names
####################
# We write scripts to advance functionally

dff = load_application_train()
dff.head()

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

# Let's leave out Passenger Id. (Exception for numeric variables)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# Let's ask if there is an Outlier
for col in num_cols:
    print(col, check_outlier(df, col))

# Let's bring the second data set and query it
cat_cols, num_cols, cat_but_car = grab_col_names(dff)

# let's disable the exception
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

# let's send the check outliers to dff
for col in num_cols:
    print(col, check_outlier(dff, col))


####################
# Accessing the Outliers Themselves
####################


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Age")
# It only got heads because it was worth more than 10.

# outliers indexes
grab_outliers(df, "Age", True)

# To save for later use
age_index = grab_outliers(df, "Age", True)

# Briefly for outliers
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

#######################
# Solving the Outlier Problem
#######################

####################
# Delete
####################

# We need upper and lower limits to identify outliers

low, up = outlier_thresholds(df, "Mouse")
df.shape

# In the reverse order of the selection made before
# Except for those below the lower limit and above the upper limit, those that are not outliers

# Number of remaining observations after deleting outliers
df[~((df["Mouse"] < low) | (df["Mouse"] > up))].shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

# Let's make it happen
for col in num_cols:
    new_df = remove_outlier(df, col)
# We got rid of contradictions with the difference between new and old data
df.shape[0] - new_df.shape[0]

####################
# Suppression Method (re-assignment with thresholds)
####################

# In suppression, values above our acceptable threshold values are replaced with threshold values.
low, up = outlier_thresholds(df, "Mouse")

df[((df["Mouse"] < low) | (df["Mouse"] > up))]["Mouse"]

df.loc[((df["Mouse"] < low) | (df["Mouse"] > up)), "Mouse"]

df.loc[(df["Mouse"] > up), "Mouse"] = up

df.loc[(df["Mouse"] < low), "Mouse"] = low


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

####################
# Recap
####################

df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")

#######################
# Multivariate Outlier Analysis: Local Outlier Factor
#############################################

#17, 3

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

# Check for outliers?
for col in df.columns:
     print(col, check_outlier(df, col))

# Let's choose one of the variables with an outlier value
low, up = outlier_thresholds(df, "carat")

# Let's see how many outliers there are
df[((df["carat"] < low) | (df["carat"] > up))].shape

# Let's choose another variable with an outlier
low, up = outlier_thresholds(df, "depth")

# Let's see how many outliers there are
df[((df["depth"] < low) | (df["depth"] > up))].shape


# With the method we imported, the number of neighborhoods is set to a default value.

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
Let's bring #LocalOutlierFactor scores

# We keep it for traceability
df_scores = clf.negative_outlier_factor_
Let's observe #5
df_scores[0:5]
# - If we do not want to observe with values
# df_scores = -df_scores
# - we will prefer to use it with negative values because
# Threshold value for a user's perspective to make the decision easier to read in the graphical technique
# Let's rank these values from smallest to largest, the five worst observations
np.sort(df_scores)[0:5]

# Let's interpret it by applying the Elbow method
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
# By thinking about what the threshold value should be, we choose the point where the slope change is most obvious. The next points are small.

# We select the value in the 3rd index
th = np.sort(df_scores)[3]

# We call values lower than the threshold values as outliers. (Observations such as -5,-6,-7 are expected)
df[df_scores < th]

df[df_scores < th].shape

# We want to understand why there are contradictions with this value. There were thousands of contradictions when done individually, but they decreased when done multiple times.
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# Analysis comment;
# We look at the first observation; we look at the Carat value, when we look at the average and max status, there seems to be no abnormal situation;
# When we look at Depth, the maximum value is 79, but 78 is not 79. The value of 79 in the Data Set has not arrived. The value of 78 has arrived.
# Table value can be average 54, maximum 95
# No change for price, x, y, z are also within expectations.
# 79 why did not come because the depth value is 78200 and the price is 1262 may have affected it.
# The depth value is 78200, the carat value is 1030 and the price cannot be 1262, or the fact that it was may have been a factor.
# We look at the second observation; we look at the Carat value, the average is reasonable, Depth is fully suitable, Table is appropriate, Price value is appropriate,
# x suitable, y suitable, Z outlier value (max value). Likewise, the relationships of the found values may have had an impact.

# Let's index
df[df_scores < th].index

# We are deleting it because it is a small number.
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# The suppression method can also be added. We will need to remove the value that creates the discrepancy and replace it with another observation.
# At this point, it is not an option as it will create a certain residue. If the number of observations is a little high, touching this place as a change will create a problem.
# If we are working with tree methods, we will shave them from the tip without touching them. We will approach the observations in themselves.


#######################
# Missing Values
#######################

#######################
# Capturing Missing Values
#######################
df = load()
df.head()

# query whether there are missing observations or not
df.isnull().values.any()

# number of missing values in variables
df.isnull().sum()

# number of exact values in variables
df.notnull().sum()

# total number of missing values in the data set
df.isnull().sum().sum()

# observation units with at least one missing value
df[df.isnull().any(axis=1)]

# observation units that are complete
df[df.notnull().all(axis=1)]

# sort in descending order
df.isnull().sum().sort_values(ascending=False)

# For the proportion of this missingness in the entire data set
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# Only the names of variables with missing values
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
     na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

     n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
     ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
     missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
     print(missing_df, end="\n")

     if na_name:
         return na_columns


missing_values_table(df)

missing_values_table(df, True)

#######################
# Solving the Missing Value Problem
#######################

missing_values_table(df)

####################
#Solution 1: Quick deletion
####################
df.dropna().shape
# When there is sufficient data size, deleting may be a solution, especially for large data.

####################
#Solution 2: Filling with Simple Assignment Methods
####################

# Let's fill it with the average of the age variable
df["Age"].fillna(df["Age"].mean()).isnull().sum()
# Let's fill it with the median of the age variable
df["Age"].fillna(df["Age"].median()).isnull().sum()
# Let's fill the age variable with a constant number
df["Age"].fillna(0).isnull().sum()

# To fill quickly. Axis 0 to move forward by looking at the rows
# df.apply(lambda x: x.fillna(x.mean()), axis=0) (It will give an error, it assigns to numeric and non-numeric values)

# To populate numeric variables only
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

# Let's save
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

# It would be correct to take the mode for categorical variables.
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# Fill with string missing expression as any special expression
df["Embarked"].fillna("missing")

# To do this automatically (to categorical variables)
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0) .isnull().sum()


####################
# Assigning Values in Categorical Variable Breakdown
####################

# Let's group by breakdown
df.groupby("Sex")["Age"].mean()

# Let's look at the average
df["Age"].mean()

# Group the age variable according to gender, select the age variable and write the average of this age variable in the relevant places in the same group breakdown.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#according to groupbya the average one is for women
df.groupby("Sex")["Age"].mean()["female"]

# To make it more clear
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"]. mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"]. mean()["male"]
# We reset the dataframe with missing value check.
df.isnull().sum()

#######################
#Solution 3: Filling with Predictive Assignment
#######################

# 1-We need to insert categorical variables into one hot encoder
# 2- Since KNN is a distance-based algorithm, we need to standardize the variables

df = load()

# We call the function to use the numerical and categorical variables we captured.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# We extract PassengerId from numerical variables.
num_cols = [col for col in num_cols if col not in "PassengerId"]
# We need to convert cat cols. In order to create a label and one hot encoder at the same time;
# We can use the get dummies method to implement one hot encoder.
# If we set the drop_first argument to True, it will drop the first class of categorical variables that have two classes and keep the second class.
# We will express categorical variables numerically.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# Here cat_cols and num_cols are two lists. We collected them.
# Get dummies method applies transformations only to categorical variables, even if you give all the variables together.
# cat_but_car does not carry any information so we left it out.
dff.head()

# Standardization of Variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# implementation of knn.
# Missing values will be filled based on prediction through machine learning.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
# With the Knn method, the average of the values close to the missing value is written here. Here we take the 5 closest values.
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# Get back the standardized values and get the values back.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# We made the assignment, but I cannot compare the old version of the Age variable with what I assigned it to. I want to follow it from this perspective.
df["age_imputed_knn"] = dff[["Age"]]

# With machine learning technique, we replaced the missing values in the data set with predicted values.
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

# Here I want to see and examine how all variables are assigned.
df.loc[df["Age"].isnull()]

####################
#Recap
####################

df = load()
# missing table
missing_values_table(df)
# filling numeric variables directly with median
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# filling categorical variables with mode
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0) .isnull().sum()
# fill numeric variables in categorical variable breakdown
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Filling with Predictive Assignment

#######################
# Advanced Analytics
#######################

####################
# Examining Missing Data Structure
####################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

# Do the missing values occur with a certain randomness?
# Do the deficiencies occur in both variables?
# No significant correlation.
msno.heatmap(df)
plt.show()

####################
# Examining the Relationship of Missing Values with the Dependent Variable
####################

#
missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
     temp_df = dataframe.copy()

     for col in na_columns:
         temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

     na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

     for col in na_flags:
         print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                             "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)

####################
#Recap
####################

df = load()
na_cols = missing_values_table(df, True)
# Filling numeric variables directly with median
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# Filling categorical variables with mode
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0) .isnull().sum()
# Filling numerical variables in the categorical variable breakdown
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Filling with Predictive Assignment
missing_vs_target(df, "Survived", na_cols)

#######################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#######################

#######################
# Label Encoding & Binary Encoding
#######################

df = load()
df.head()
df["Sex"].head()

# When we want to code the sex variable as 0 and 1
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
# To remember which is 0 and which is 1
le.inverse_transform([0, 1])

# A two-class operation with function dataframe and binary_col
def label_encoder(dataframe, binary_col):
     labelencoder = LabelEncoder()
     dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
     return dataframe

df = load()

# If we have more than one variable, we choose binary_cols.
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                and df[col].nunique() == 2]

for col in binary_cols:
     label_encoder(df, col)

df.head()

# Let's add a larger data set
df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                and df[col].nunique() == 2]

df[binary_cols].head()


for col in binary_cols:
     label_encoder(df, col)


df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())


#######################
# One-Hot Encoding
#######################

# It is important to drop the first class.

df = load()
df.head()
df["Embarked"].value_counts()

# Transform Embarked class into dummie
pd.get_dummies(df, columns=["Embarked"]).head()

# To avoid falling into the dummy trap, we get rid of the first class by considering that the variables cannot be generated from each other.
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# If we want the values in the relevant variable to come as a class; the missing values class is created
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# To do it all in one step (for one-hot encoder and label encoder).
# Two-class categorical variables are expressed as 1-0 without the need to apply a label encoder.
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

# Let's functionalize
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

df.head()

#######################
# Rare Encoding
#######################

# 1. Analyzing the abundance of categorical variables.
# 2. Analyzing the relationship between rare categories and the dependent variable.
# 3. We will write rare encoder.
#4. When there is a data set with plenty of categorical variables we should use Rare Analyzer and at least know
# - It is necessary to know which categorical variable's class, which frequency, which rate-dependent variable has an effect in terms of target.
####################
# 1. Analyzing the abundance of categorical variables.
####################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

# We select categorical variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
     print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
     print("####################################################"
     if plot:
         sns.countplot(x=dataframe[col_name], data=dataframe)
         plt.show()


for col in cat_cols:
     cat_summary(df, col)

####################
# 2. Analyzing the relationship between rare categories and the dependent variable.
####################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, cat_cols):
     for col in cat_cols:
         print(col, ":", len(dataframe[col].value_counts()))
         print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                             "RATIO": dataframe[col].value_counts() / len(dataframe),
                             "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)


#######################
#3. Writing the rare encoder.
#######################
# Our Rare function collects and brings together the sparse classes of categorical variables with sparse classes in the data set and names them Rare.
def rare_encoder(dataframe, rare_perc):
     temp_df = dataframe.copy()

     rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                     and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

     for var in rare_columns:
         tmp = temp_df[var].value_counts() / len(temp_df)
         rare_labels = tmp[tmp < rare_perc].index
         temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

     return temp_df

# We entered Df and gave it a rare rate of 0.01.
# It will bring together the categorical variable classes that fall below this rate.
new_df = rare_encoder(df, 0.01)


rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#######################
# Feature Scaling
#######################

####################
# StandardScaler: Classic standardization. Subtract the mean, divide by the standard deviation. z = (x - u) / s
####################

# To eliminate measurement differences between variables.
# To ensure that the models to be used approach the variables equally.
# To shorten the training time of algorithms using gradient descent
# In distance-based methods, variables with large values exhibit dominance.

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

####################
# RobustScaler: Subtract median and divide by iqr.
####################
# Standard Scaler subtracts the mean from all observation units and divides it by the standard deviation.
# Standard deviation and mean are values affected by the metrics in the data set.
# Therefore, if we subtract the Median from all observation units,
# Then, for division, use the value that is affected by the outliers, such as the standard deviation, and not the value that is affected by the outliers.
If we divide it by # iqr, we will take into account both the central tendency and change and make a more robust standardization.
#
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T
# There is a change in the representation point, not in the distribution structure.
# Also, the data will not be corrupted. The mean and median of the Robust scaler are in a different position than the standard scaler, the reason for this is that it is not affected by outliers.

####################
# MinMaxScaler: Variable conversion between 2 given values
####################
# It can be used especially for special ranges that you want to convert, such as 0-1 1-5 0-10.
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T
# I expected min 0 max:1 and the distribution between the two was as expected

df.head()

age_cols = [col for col in df.columns if "Age" in col]

# To show quarters of a numerical variable and create its graph.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)

####################
# Numeric to Categorical: Converting Numeric Variables to Categorical Variables
# Binning
####################
# The qcut function sorts the values of a variable from smallest to largest and divides them into 5 parts.
df["Age_qcut"] = pd.qcut(df['Age'], 5)

#######################
# Feature Extraction
#######################

#######################
# Binary Features: Flag, Bool, True-False
#######################

df = load()
df.head()
# Let's create a new set of empty values in response to the empty variables in the cabinets
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

# Let's take the average according to our dependent variable
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
# Those with occupied cabins have a higher survival rate than those with empty cabins

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# There is no difference between the P1 P2 ratios. It expresses the survival or non-survival status of the two groups.
# The h0 hypothesis, which states that there is no difference between the two, is rejected because the p value is less than 0.05.
# There is a statistically significant difference between them.

# Let's create a binary future. Let's look at it as relatives on the ship.
# Bring two variables together, put a condition and create a new variable according to this condition.
# In terms of survival reflex formation depending on whether the person is alone on the ship or not

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

# Let's evaluate the operation in terms of survival.
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})
# There seems to be a difference between them. Those with a family have a higher survival rate.
# It may have occurred by chance, or there may be other multivariate effects.

# Let's do a hypothesis test
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# The h0 hypothesis, which states that there is no difference between the two, is rejected because the p value is less than 0.05.
# There is a statistically significant difference between them.

#######################
# Deriving Features from Texts
#######################

df.head()

####################
# LetterCount
####################
df["NEW_NAME_COUNT"] = df["Name"].str.len()
####################
# word count
####################
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
####################
# Capturing Special Structures
####################
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

####################
# Deriving Variables with Regex
####################

df.head()
# Let's create a pattern
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#######################
# Generating Date Variables
#######################

dff = pd.read_csv("Feature_Engineering/datasets/course_reviews.csv")
dff.head()
dff.info()
# Variable type conversion to generate variable from Timestamp variable
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (month difference between two dates): year difference + month difference
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

#dayname
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

#date

#######################
# Feature Interactions
#######################

df = load()
df.head()
# Such as multiplying variables by adding them together and squaring them.
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()
# Does this new variable mean anything?
df.groupby("NEW_SEX_CAT")["Survived"].mean()

#######################
# Titanic End-to-End Feature Engineering & Data Preprocessing
#######################

df = load()
df.shape
df.head()
# We make variable names larger
df.columns = [col.upper() for col in df.columns]

#######################
# 1. Feature Engineering (Variable Engineering)
#######################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
#name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
#name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
#name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
#familysize
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
#age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
#isalone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
#age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Passenger Id is not a numerical variable, we leave it out
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#######################
#2. Outliers
#######################

for col in num_cols:
     print(col, check_outlier(df, col))

for col in num_cols:
     replace_with_thresholds(df, col)

for col in num_cols:
     print(col, check_outlier(df, col))

#######################
# 3. Missing Values
#######################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# Let's take a new title gore groupby and fill in the missing values of the age variable with the median new title gore
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# We recreate all the variables we created based on age
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# For Embareked, we will fill the categorical variables whose type is object and whose number of unique values is less than 10, with their modes.
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#######################
#4. Label Encoding
#######################
# We have converted two-class categorical variables. First, let's select
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                and df[col].nunique() == 2]

for col in binary_cols:
     df = label_encoder(df, col)


#######################
#5. Rare Encoding
#######################
# First of all, we will do Rare Encoding for possible reductions instead of One hot encoding

rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#######################
#6. One-Hot Encoding
#######################

# We chose variables that have more than 2 unique values and variables that have less than 10 unique values
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

# To capture unusable columns
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                 (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# We can delete it if we want
# df.drop(useless_cols, axis=1, inplace=True)

#######################
#7. Standard Scaler
#######################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#######################
#8. Model
#######################
# Dependent variable Survived, independent variables Paasnger id and variables other than Surived
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)
# We divide the data set into two: train and test.
# We will hold a model in Train, we will be testing this model that I built with the test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Let's bring the model object, a tree-based method
from sklearn.ensemble import RandomForestClassifier

# Setting up the model x train independent variables y train target dependent variable
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
# Prediction of dependent variables of the test set
# Set the test set with the dependent variable y
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#######################
# The score to be obtained without any action?
#######################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# How are the newly produced variables doing?

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










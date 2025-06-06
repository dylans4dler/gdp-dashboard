import streamlit as st
import pandas as pd
import math
from pathlib import Path
from snowflake.snowpark.context import get_active_session
import cartopy
import warnings
import snowflake.connector
# Libraries to help with reading and manipulating data
import numpy as np
# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)



# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Library to split data
from sklearn.model_selection import train_test_split

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    #plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)


#from streamlit_carousel import carousel

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data

def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)
    

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# import data------------------------------------------------------
session = get_active_session()

def load_data(table_name):
    ## Read in data table
    st.write(f"Here's some example data from `{table_name}`:")
    table = session.table(table_name)
    
    ## Do some computation on it
    table = table.limit(100)
    
    ## Collect the results. This will run the query and download the data
    table = table.collect()
    return table

# Select and display data table

# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: This is test Blanc

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

#------------------------------------------------

table_name = "RESORT.PUBLIC.RESORT"
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQKT66qWepnKo8tbPDzvNVwBY61wNRz0ddjqu3BlSEtQaRTyvrVaKgBGx4IOmAfbVSVlaatL-6HzE-I/pub?gid=1431853571&single=true&output=csv"
sql = 'select * from RESORT.PUBLIC.RESORT' 
#------------------------------------------------
sql = 'select * from RESORT.PUBLIC.RESORT'    
data = session.sql(sql).to_pandas()

#pd.read_csv(result)
st.image('PIE.png')
#st.pyplot(data)

#make functions ---------------------------------------------------
def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

    # function to create labeled barplots

#------------------------------------------------

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot
    
#---------------------------------------------
    
def correlation(data):
    cols_list = data.select_dtypes(include=np.number).columns.tolist()

    plt.figure(figsize=(12, 7))
    sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
    )
    plt.show()
    
#----------------------------------------------------
    
def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()
#----------------------------------------------------------

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
    
#-----------------------------------------------------------
def boxplt():
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data, x="BOOKING_SOURCE", y="TOTAL_PAID", palette="gist_rainbow"
        )
    plt.show()

#----------------------------------------------------------------
#still bug
def topmonth():
    df = data.copy()
    df['ARRIVAL_MONTH'] = ''
    for idx, r in enumerate(df['ARRIVAL_DATE'].to_string):
        df['ARRIVAL_MONTH'].iloc[idx] = int(r[1:10]) #bug here

    data['ARRIVAL_MONTH'] = df['ARRIVAL_MONTH'].astype(int)

    # grouping the data on arrival months and extracting the count of bookings
    monthly_data = data.groupby(["ARRIVAL_MONTH"])["STATUS"].count()

    # creating a dataframe with months and count of customers in each month
    monthly_data = pd.DataFrame(
        {"Month": list(monthly_data.index), "Guests": list(monthly_data.values)}
    )

    # plotting the trend over different months
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_data, x="Month", y="Guests")
    plt.show()

#outlier detection using boxplot-----------------------------------------
def outlier():
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    # dropping booking_status
    numeric_columns.remove("STATUS")

    plt.figure(figsize=(15, 12))

    for i, variable in enumerate(numeric_columns):
        plt.subplot(4, 4, i + 1)
        plt.boxplot(data[variable], whis=1.5)
        plt.tight_layout()
        plt.title(variable)

    plt.show()
    
#-------------------------------------------------------------------
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf

# defining a function to plot the confusion_matrix of a classification model
def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

#--------------------------------------------------------------------

def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(len(predictors.columns))
    ]
    return vif
#---------------------------------------------------------------------

def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf

#----------------------------------------------------------------------

def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

#--------------------------------------------------------------------
def feature_import():
    feature_names = list(X_train.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(8, 8))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()



#display data --------------------------------------------------------
st.dataframe(data)
st.pyplot(histogram_boxplot(data, "LEAD_TIME"))
st.pyplot(histogram_boxplot(data,"TOTAL_PAID"))
st.pyplot(labeled_barplot(data, "NIGHTS", perc=True))
st.pyplot(labeled_barplot(data,"ROOM_NUMBER",perc=True))
st.pyplot(labeled_barplot(data,"BOOKING_SOURCE",perc=True))
    
st.pyplot(correlation(data))
st.pyplot(boxplt())
st.pyplot(stacked_barplot(data, "BOOKING_SOURCE", "STATUS"))
st.pyplot(distribution_plot_wrt_target(data, "TOTAL_PAID", "STATUS"))
st.pyplot(stacked_barplot(data, "NIGHTS", "STATUS"))
#st.pyplot(topmonth())
st.pyplot(outlier())


#--------modelinig-------------------------------------------
X = data.drop(["STATUS","GUEST_NAME","OUR_REFERENCE","ARRIVAL_DATE","DEPARTURE_DATE","BOOKING_DATE","ROOM_NUMBER","BOOKING_SOURCE","NIGHTS","AMOUNT_OF_ROOMS"], axis=1)
Y = data["STATUS"]

# adding constant
X = sm.add_constant(X) ## Complete the code to add constant to X

X = pd.get_dummies(X,drop_first=True) ## Complete the code to create dummies for X

X

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1) 
## Complete the code to split the data into train test in the ratio 70:30 with random_state = 1
st.write(f'Shape of Training set :  {X_train.shape} ')
st.write(f'Shape of test set :  {X_test.shape}')
st.write("Percentage of classes in training set:")
st.write(y_train.value_counts(normalize=True))
st.write("Percentage of classes in test set:")
st.write(y_test.value_counts(normalize=True))

# fitting logistic regression model----------------------------------------
#logit = sm.Logit(y_train, X_train.astype(float))
#lg = logit.fit() 
## Complete the code to fit logistic regression---------------------------

#print(lg.summary()) ## Complete the code to print summary of the model

#checking_vif(X_train)

# initial list of columns
cols = X_train.columns.tolist()

# setting an initial max p-value
max_p_value = 1

#while len(cols) > 0:
#    # defining the train set
#    x_train_aux = X_train[cols]
#
#    # fitting the model
#    model = sm.Logit(y_train, x_train_aux).fit(disp=False)
#
#    # getting the p-values and the maximum p-value
#    p_values = model.pvalues
#    max_p_value = max(p_values)
#
#    # name of the variable with maximum p-value
#    feature_with_p_max = p_values.idxmax()
#
#    if max_p_value > 0.05:
#        cols.remove(feature_with_p_max)
#    else:
#        break
#
#selected_features = cols
#print(selected_features)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train) 
## Complete the code to fit decision tree on train data
st.pyplot(confusion_matrix_sklearn(model, X_train, y_train)) 
## Complete the code to create confusion matrix for train data

decision_tree_perf_train = model_performance_classification_sklearn(
    model, X_train, y_train
)
decision_tree_perf_train

confusion_matrix_sklearn(model, X_test, y_test) 
## Complete the code to create confusion matrix for test data

decision_tree_perf_test = model_performance_classification_sklearn(model, X_test, y_test) ## Complete the code to check performance on test set
decision_tree_perf_test

st.pyplot(feature_import())


#pruning------------------------
#pre-pruning------------------------

# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1, class_weight="balanced")

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(2, 7, 2),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(f1_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)

st.pyplot(confusion_matrix_sklearn(estimator, X_train, y_train)) 
## Complete the code to create confusion matrix for train data

decision_tree_tune_perf_train = model_performance_classification_sklearn(estimator, X_train, y_train) ## Complete the code to check performance on train set
decision_tree_tune_perf_train

st.pyplot(confusion_matrix_sklearn(estimator, X_test, y_test)) 
## Complete the code to create confusion matrix for test data

decision_tree_tune_perf_test = model_performance_classification_sklearn(estimator, X_test, y_test) ## Complete the code to check performance on test set
decision_tree_tune_perf_test

def decision_tree():
    plt.figure(figsize=(20, 10))
    out = tree.plot_tree(
        estimator,
        #feature_names=feature_names,
        filled=True,
        fontsize=9,
        node_ids=False,
        class_names=None,
    )
    # below code will add arrows to the decision tree split if they are missing
    for o in out:
        arrow = o.arrow_patch
        if arrow is not None:
            arrow.set_edgecolor("black")
            arrow.set_linewidth(1)
    plt.show()
    
st.pyplot(decision_tree())

# Text report showing the rules of a decision tree -
st.write(tree.export_text(estimator, show_weights=True))

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)))
plt.xlabel("Relative Importance")
st.pyplot(plt.show())

clf = DecisionTreeClassifier(random_state=1, class_weight="balanced")
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = abs(path.ccp_alphas), path.impurities

st.dataframe(path)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
st.pyplot(plt.show())

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=1, ccp_alpha=ccp_alpha, class_weight="balanced"
    )
    clf.fit(X_train, y_train) ## Complete the code to fit decision tree on training data
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
st.pyplot(fig.tight_layout())

f1_train = []
for clf in clfs:
    pred_train = clf.predict(X_train)
    values_train = f1_score(y_train, pred_train)
    f1_train.append(values_train)

f1_test = []
for clf in clfs:
    pred_test = clf.predict(X_test)
    values_test = f1_score(y_test, pred_test)
    f1_test.append(values_test)

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs alpha for training and testing sets")
ax.plot(ccp_alphas, f1_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, f1_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
st.pyplot(plt.show())

index_best_model = np.argmax(f1_test)
best_model = clfs[index_best_model]
st.write(best_model)

st.pyplot(confusion_matrix_sklearn(best_model, X_train, y_train))

decision_tree_post_perf_train = model_performance_classification_sklearn(
    best_model, X_train, y_train
)
decision_tree_post_perf_train

st.pyplot(confusion_matrix_sklearn(best_model, X_train, y_train))
## Complete the code to create confusion matrix for test data on best model

decision_tree_post_test = model_performance_classification_sklearn(
    best_model, X_test, y_test
) ## Complete the code to check performance of test set on best model
decision_tree_post_test

plt.figure(figsize=(20, 10))

out = tree.plot_tree(
    best_model,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
st.pyplot(plt.show())

# Text report showing the rules of a decision tree -
st.write(tree.export_text(best_model, show_weights=True))

importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)))
plt.xlabel("Relative Importance")
st.pyplot(plt.show())

# training performance comparison

models_train_comp_df = pd.concat(
    [
        decision_tree_perf_train.T,
        decision_tree_tune_perf_train.T,
        decision_tree_post_perf_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
st.write("Training performance comparison:")
models_train_comp_df

# testing performance comparison

models_test_comp_df = pd.concat(
    [
        decision_tree_perf_test.T,
        decision_tree_tune_perf_test.T,
        decision_tree_post_test.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Testing performance comparison:")
models_test_comp_df ## Complete the code to compare performance of test set
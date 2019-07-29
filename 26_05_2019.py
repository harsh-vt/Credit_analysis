import pandas as pd # used for data structure
import numpy as np # used for data structure
import seaborn as sns # plot lib
import matplotlib.pyplot as plt # plot lib
from collections import Counter # Dict subclass for counting hashable items
train_input = pd.read_csv("./data/CreditRiskTrainData.csv")
test_input = pd.read_csv("./data/CreditRiskTestData.csv")
# print column
print (train_input.columns)
print (test_input.columns)
# outlier treatment
train_input.describe().head() # gives summary of data
#train_input["ApplicantIncome"].hist(bins=50) 
#train_input.boxplot(column="ApplicantIncome")
#train_input.boxplot(column="ApplicantIncome", by = "Education")
#train_input["LoanAmount"].hist(bins=50)
#train_input.boxplot(column="LoanAmount")
# missing value fix
train_input.apply(lambda x: sum(x.isnull()),axis = 0) # displays na in each column
train_input["LoanAmount"].fillna(train_input["LoanAmount"].mean(), inplace = True) # replace empty loan amount to mean
train_input.apply(lambda x: sum(x.isnull()),axis = 0)
train_input["Self_Employed"].value_counts()
train_input["Self_Employed"].fillna("No",inplace = True) # fills self employed na to no
train_input.apply(lambda x: sum(x.isnull()),axis = 0)
train_input["Gender"].fillna(train_input["Gender"].mode()[0], inplace = True) # fills gender na to mode of column. [0] for allowing one value if mode return list of 2
train_input["Married"].fillna(train_input["Married"].mode()[0], inplace = True)
train_input["Dependents"].fillna(train_input["Dependents"].mode()[0], inplace = True)
train_input["Loan_Amount_Term"].fillna(train_input["Loan_Amount_Term"].mode()[0], inplace = True)
train_input["Credit_History"].fillna(train_input["Credit_History"].mode()[0], inplace = True)
train_input.apply(lambda x: sum(x.isnull()),axis = 0)
# cotegorical var analysis
temp1 = train_input["Credit_History"].value_counts(ascending = True)
print ("Frequency Table For Credit History: ")
print (temp1)
temp2 = pd.crosstab(train_input["Credit_History"], train_input["Loan_Status"]) # plots graph showing credit history over loan status
print (temp2)
# decriptive predictive prescriptive
#temp2.plot(kind = "bar", stacked = True, color = ["red", "blue"], grid = False)
def percConvert(ser):
    return ser/float(ser[-1]) # function for checking probability of loan status over each credit history
pd.crosstab(train_input["Credit_History"], train_input["Loan_Status"], margins = True).apply(percConvert, axis = 1)
train_input.head()
from sklearn.preprocessing import LabelEncoder
var_mod = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
le = LabelEncoder() # used for labeling data as ordinals
for i in var_mod:
    train_input[i] = le.fit_transform(train_input[i]) 
train_input.dtypes
train_input.head()
y_train = train_input["Loan_Status"]
y_train.head()
x_train = train_input.drop(["Loan_Status", "Loan_ID"], axis = 1)
x_train.head()
test_input.apply(lambda x: sum(x.isnull()), axis = 0)
test_input["Gender"].fillna(test_input["Gender"].mode()[0], inplace = True)
test_input["LoanAmount"].fillna(test_input["LoanAmount"].mean(), inplace = True)
test_input["Dependents"].fillna(test_input["Dependents"].mode()[0], inplace = True)
test_input["Self_Employed"].fillna(test_input["Self_Employed"].mode()[0], inplace = True)
test_input["Property_Area"].fillna(test_input["Property_Area"].mode()[0], inplace = True)
test_input["Loan_Amount_Term"].fillna(test_input["Loan_Amount_Term"].mode()[0], inplace = True)
test_input["LoanAmount"].fillna(test_input["LoanAmount"].mean(), inplace = True)
test_input["Credit_History"].fillna(test_input["Credit_History"].mode()[0], inplace = True)
test_input.head()
test_input.apply(lambda x: sum(x.isnull()),axis = 0)
x_test = test_input.drop(["Loan_ID"], axis = 1) # coding test data
var_mod = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
le = LabelEncoder()
for i in var_mod:
    test_input[i] = le.fit_transform(test_input[i])
test_input.dtypes
test_input.head()
from sklearn.linear_model import LogisticRegression # for logistic regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print("Acuracy of logistic regression classifier on test data: {:.2f}".format(log_reg.score(x_train, y_train))) # confusion matrix
y_pred = log_reg.predict(x_train)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train, y_pred)
print (confusion_matrix)
y_test = log_reg.predict(x_test) # starting to predict values
print (y_test)
# for ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_train, log_reg.predict(x_train))
fpr, tpr, thresholds = roc_curve(y_train, log_reg.predict_proba(x_train)[:,1])
plt.Figure()
plt.plot(fpr, tpr, label = "Logistic Regression (area = %0.2f)" % logit_roc_auc)
plt.plot([0,1],[0,1],"r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Reciever operating characteristics")
plt.legend(loc = "Lower right")
plt.savefig("log_ROC")
plt.show()
# PREDICT PROBABILITY
y_train_probs = log_reg.predict_proba(x_train)
# changing threshold of fraud form 50
prob = y_train_probs[:,1]
prob_df = pd.DataFrame(prob)
prob_df["predict"] = np.where(prob_df[0] >= 0.80, 1, 0)
prob_df.head()
from sklearn import metrics
print (metrics.accuracy_score(y_train, prob_df["predict"]))
confusion_matrix = pd.crosstab(y_train, prob_df["predict"])
print (confusion_matrix)
# cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(log_reg, x_train, y_train, scoring = "accuracy", cv = 8)
print (scores)
print (scores.mean())


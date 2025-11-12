# z5270589 HW1 Question1 and Question2
# Orignially I finished my work on google colab, now I moved it to local and each time it only can show one picture
# you should close the first picture and the second picture can come out.
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import *
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV

#drive.mount('/content/drive')
# I use google colab in this assignemnt, firstly I'll connect the google server and then I read the csv file.
#initial_data = pd.read_csv('/content/drive/My Drive/9417/heart.csv')
# Now I move it to my local device, I can read the csv in the folder.
initial_data = pd.read_csv('./heart.csv')
pd.set_option("display.max_rows", 100)
#display(initial_data)

# The Diagrams created by step(a) to step(g)
def Hist_Age_formation(X):
    sns.histplot(X["Age"], bins=40, kde=True, color="black")
    plt.xlabel("Age")
    plt.ylabel("Age Count")
    plt.title("(b)Fixed Age Distribution Diagram")
    plt.show()

def Hist_Gender_Smoke_formation(X):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(X["Gender"], bins=40, kde=True, color="black")
    plt.xlabel("Gender")
    plt.ylabel("Gender Count")
    plt.title("(c)Fixed Gender Distribution Diagram")
    #output image:
    plt.subplot(1, 2, 2)
    sns.histplot(X["Smoker"], bins=20, kde=True, color="black")
    plt.xlabel("Smoker")
    plt.ylabel("Smoker Count")
    plt.title("(c)Fixed smoke Distribution Diagram")

    plt.show()

def Systolic_diastolic(X):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(X["Systolic"], bins=40, kde=True, color="black")
    plt.xlabel("Systolic")
    plt.ylabel("Systolic Count")
    plt.title("(d)Systolic Distribution Diagram")
    #output image:
    plt.subplot(1, 2, 2)
    sns.histplot(X["Diastolic"], bins=20, kde=True, color="black")
    plt.xlabel("Diastolic")
    plt.ylabel("Diastolic Count")
    plt.title("(d)Diastolic Distribution Diagram")

    plt.show()

# Question 1. Data Wrangling #

# (a)
# Variably y containing the target (Heart Disease).
y = initial_data['Heart_Disease']
# Remove the Last Checkup feature, X containing only features
X = initial_data.drop(columns=['Last_Checkup', 'Heart_Disease'])
#
# display(y.head())
# display(X.head())
# display(X.describe())
# display(y.describe())
print(f"\033[1mQ1(a):\033[0m")
X.info()
y.info()
# X.isna().sum()

# (b)
# Replaced age with positive versions.
X['Age'] = X['Age'].abs()

#display(X['Age'].head())
print(f"\033[1mQ1(b):\033[0m")
display(X.iloc[31]['Age'])
display(X.describe())

# Diagram for (b)
Hist_Age_formation(X)
print(X['Gender'].dtypes)
print(X['Smoker'].dtypes)

# (c)
gender_replace_list = {'Male': 0, 'M': 0, 'Female': 1, 'F': 1, 'Unknown': 2} # map (Male/M,Female/F, Unknown) to (0,1,2).
smoker_replace_list = {'No': 0, 'N': 0, 'Yes': 1, 'Y': 1, np.nan: 2} # For Smoker, map (No/N,Yes/Y,Nan) to (0,1,2).
X['Gender'] = X['Gender'].replace(gender_replace_list).infer_objects(copy=False).astype(int)
X['Smoker'] = X['Smoker'].replace(smoker_replace_list).infer_objects(copy=False).astype(int)
# Future warning that `replace` is deprecated and will be removed in a future version, to retain the old behavior, explicitly call `result.infer_objects(copy=False).
print(f"\033[1mQ1(c):\033[0m")
print(X['Gender'].dtypes)
print(X['Smoker'].dtypes)
print(X)

# Diagram for (c)
print(X['Smoker'].isna().sum()) # check wether the nan has been transfered
Hist_Gender_Smoke_formation(X)

# (d)
# Use expand=True here to create create two new columns, which represents systolic and diastolic.
X_Blood_Pressure_Split = X['Blood_Pressure'].str.split('/', expand=True).astype(float)
X['Systolic'] = X_Blood_Pressure_Split[0]
X['Diastolic'] = X_Blood_Pressure_Split[1]
# Remove the original blood pressure variable.
X.drop(columns=['Blood_Pressure'], inplace=True)
# same as X = X.drop(columns=['Blood_Pressure']), use inplace=True here can directly change X.

print(f"\033[1mQ1(d):\033[0m")
# Diagram for (d)
X.info()
Systolic_diastolic(X)

# (e)
# Split the data into training and test size parameter to 0.3, and the random state to ‘2’
X_trainingSet, X_testSet, y_trainingSet, y_testSet = train_test_split(X, y, test_size=0.3, random_state=2)

# Diagram for (e)
print(f"\033[1mQ1(e):\033[0m")
print("\033[1m(e)The information of X training Set :\033[0m")
X_trainingSet.info()
print("\n")
print("\033[1m(e)The information of X test Set :\033[0m")
X_testSet.info()
print("\n")
print("\033[1m(e)The information of y training Set :\033[0m")
y_trainingSet.info()
print("\n")
print("\033[1m(e)The information of y test Set :\033[0m")
y_testSet.info()

# (f)
# use loc to filter that Male and his age is not Nan, then gets the median male age value based on this.
median_male_age = X_trainingSet.loc[(X_trainingSet['Gender'] == 0) & (~X_trainingSet['Age'].isna()), 'Age'].median()
# same, gets the median female age value based on this
median_female_age = X_trainingSet.loc[(X_trainingSet['Gender'] == 1) & (~X_trainingSet['Age'].isna()), 'Age'].median()

# Change the Nan age of males in the training and test set.
X_trainingSet.loc[(X_trainingSet['Gender'] == 0) & (X_trainingSet['Age'].isna()), 'Age'] = median_male_age
X_testSet.loc[(X_testSet['Gender'] == 0) & (X_testSet['Age'].isna()), 'Age'] = median_male_age

# Change the Nan age of females in the training and test set.
X_trainingSet.loc[(X_trainingSet['Gender'] == 1) & (X_trainingSet['Age'].isna()), 'Age'] = median_female_age
X_testSet.loc[(X_testSet['Gender'] == 1) & (X_testSet['Age'].isna()), 'Age'] = median_female_age
# made a copy of X_trainingSet, can used in the future
X_trainingSet_original = X_trainingSet.copy()

# Diagram for (f)
print(f"\033[1mQ1(f):\033[0m")
print("\033[1mNumbers of 'Age' values missing in my original X data: \033[0m", X['Age'].isna().sum())
print("\033[1mNumbers of 'Age' values missing in my test data: \033[0m", X_trainingSet['Age'].isna().sum())
print("\033[1m(f)The information of X training Set :\033[0m")
X_trainingSet.info()
print("\n")
print("\033[1m(f)The information of X test Set :\033[0m")
X_testSet.info()
print("\n")
print("\033[1m(f)The information of y training Set :\033[0m")
y_trainingSet.info()
print("\n")
print("\033[1m(f)The information of y test Set :\033[0m")
y_testSet.info()

# (g)
#  Scale the columns: ’Age’, ’Height feet’, ’Weight kg’, ’Cholesterol’, ’Systolic’, ’Diastolic’ using a min-max normalizer
# We can acheive this by the MinMaxScaler function.
Target_Scale_Columns = ['Age', 'Height_feet', 'Weight_kg', 'Cholesterol', 'Systolic', 'Diastolic']
X_min_max_normalizer = MinMaxScaler()
# For X training set, use fit_transform to compute the min and max:
X_trainingSet[Target_Scale_Columns] = X_min_max_normalizer.fit_transform(X_trainingSet[Target_Scale_Columns])
# For X test Set, we do also use fit_transform here to make sure to do this separately for train and test data.
X_testSet[Target_Scale_Columns] = X_min_max_normalizer.fit_transform(X_testSet[Target_Scale_Columns])

# Diagram for (g)
print(f"\033[1mQ1(g):\033[0m")
print("\033[1mOriginal X_Training set before using a min-max normalizer\033[0m")
display(X_trainingSet_original.describe())
print("\033[1mCurrent X_Training set after using a min-max normalizer\033[0m")
display(X_trainingSet.describe())

# (h)Plot a histogram of your target variable (from your training data).
sns.histplot(y_trainingSet, bins=30, kde=True, color="red")
plt.xlabel("Heart Disease Possibility")
plt.ylabel("Heart Disease Count")
plt.title("(h)Heart Disease Histogram")
plt.show()

# (h)
# Create a new target variable by quantizing the original target variable. You can do this by setting values below a certain threshold (say 0.1) to be 0 and those above the threshold to be 1.
y_trainingSet_quantizing = (y_trainingSet>0.1).astype(int)
y_testSet_quantizing = (y_testSet>0.1).astype(int)

# Histograms for the Quantized Y TrainingSet and Quantized Y TestSet:
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(y_trainingSet_quantizing, bins=5, kde=False, color="red")
plt.xlabel("Heart Disease Possibility")
plt.ylabel("Heart Disease Numbers")
plt.title("(h)Quantized Y TrainingSet")
plt.subplot(1, 2, 2)
sns.histplot(y_testSet_quantizing, bins=5, kde=False, color="red")
plt.xlabel("Heart Disease Possibility")
plt.ylabel("Heart Disease Numbers")
plt.title("(h)Quantized Y TestSet")

plt.show()

# (b)
# Create a grid of 100 C values
C_100_values = np.logspace(-4, 4, 100)
Test_log_losses, Training_log_losses = [], []

# For each C, fit a logistic regression model (using the LogisticRegression class in sklearn) on the training data.
for C in C_100_values:
    model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=200)
    model.fit(X_trainingSet, y_trainingSet_quantizing)
    # use predict proba to generate predictions from your fitted models to plug into the log-loss
    y_training_proba = model.predict_proba(X_trainingSet)[:, 1]
    y_test_proba = model.predict_proba(X_testSet)[:, 1]

    # get the log loss value
    train_log_losses = log_loss(y_trainingSet_quantizing, y_training_proba)
    test_log_losses = log_loss(y_testSet_quantizing, y_test_proba)

    Training_log_losses.append(train_log_losses)
    Test_log_losses.append(test_log_losses)

# Plot the two log_loss curves(Training/Test):
plt.figure(figsize=(10, 6))
plt.semilogx(C_100_values, Training_log_losses, label='Training Log_Loss', marker='x')
plt.semilogx(C_100_values, Test_log_losses, label='Test Log_Loss', marker='x')

plt.xlabel('C Value')
plt.ylabel('Log Loss')
plt.title('Q2(b)Training and Test Log-Loss Versus C Value')
plt.legend()
plt.grid(True)
plt.show()

# (c)
# Test_fold_log_losses to record all log-loss value.
Test_fold_log_losses = []
# We will split the train data into 5 folds.
fold_num = 5

for C in C_100_values:
    # iter_Test_fold_log_losses used to record the log-loss (5 folds)of each C value
    iter_Test_fold_log_losses = []

    # 5 iterations, each part will be treated as the testset once.
    for i in range(fold_num):

        # begin_index and terminate_index used as the pointers to find the current train and test set.
        begin_index = i * (len(X_trainingSet) // fold_num)
        terminate_index = (i + 1) * len(X_trainingSet) // fold_num if i < (fold_num - 1) else len(X_trainingSet)

        # First get the X and y test set.
        X_TestSet_fold = X_trainingSet.iloc[begin_index:terminate_index]
        y_TestSet_quantizing_fold = y_trainingSet_quantizing.iloc[begin_index:terminate_index]

        # Then the X and y training set.
        X_TrainingSet_fold = pd.concat([X_trainingSet.iloc[:begin_index], X_trainingSet.iloc[terminate_index:]])
        y_TrainingSet_quantizing_fold = pd.concat([y_trainingSet_quantizing.iloc[:begin_index], y_trainingSet_quantizing.iloc[terminate_index:]])

        # fit a logistic regression model (using the LogisticRegression class in sklearn) on the splited training data
        model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=200)
        model.fit(X_TrainingSet_fold, y_TrainingSet_quantizing_fold)

        # use predict proba to generate predictions from your fitted models to plug into the log-loss
        y_test_fold_proba = model.predict_proba(X_TestSet_fold)[:, 1]

        # get the log loss value
        test_fold_log_losses = log_loss(y_TestSet_quantizing_fold, y_test_fold_proba)
        iter_Test_fold_log_losses.append(test_fold_log_losses)

    # Record the different 5 training set log-loss value
    Test_fold_log_losses.append(iter_Test_fold_log_losses)

# we should match the Log-Loss valuae with its corresponding C value
plt.figure(figsize=(10, 6))

# plot a box-plot over the 5 CV scores.
# we use np.log10(C_100_values) here to prevent from most of values in the left side, now it will distribute average in the x axis.
plt.boxplot(Test_fold_log_losses, positions=np.log10(C_100_values), widths=0.05)

# Since we turn C into lnC before, we need re-define the X axsis to make the box plot looks better.
Refine_X = np.arange(-4, 5, 1)
# We choose 9 keypoints in the X axis, which is 10^-4-10^4
plt.xticks(Refine_X, [f"$10^{{{x_scale}}}$" for x_scale in Refine_X])
plt.xlabel('C Value')
plt.ylabel('Log Loss Value')
plt.title('Q2(c)5-Fold Cross Validation Log-Loss Boxplot')
plt.show()

# (c)
# Report the value of C that gives you the best CV performance in terms of log-loss:
# We calculate the mean log-loss for each(100) iter_Test_fold_log_losses
mean_log_losses = [np.mean(log_loss) for log_loss in Test_fold_log_losses]
# Then we can select the minimized mean log loss index among 100 5 folds mean log loss values.
best_C_index = np.argmin(mean_log_losses)
# This index is the C index that leads to the minimized mean mean log loss in the 5 folds, we can recognized it as the best C
Best_C = C_100_values[best_C_index]
print(f"\033[1mQ2(c):\033[0m")
print(f"\033[1mQ2(c)Best C value: {Best_C}\033[0m")

# Re-fit the model with this chosen C:
model = LogisticRegression(C=Best_C, penalty='l2', solver='lbfgs', max_iter=200)
model.fit(X_TrainingSet_fold, y_TrainingSet_quantizing_fold)

best_y_train_pred = model.predict(X_trainingSet)
best_y_test_pred = model.predict(X_testSet)

# Report both train and test accuracy using this model
# we use the function accuracy_score() to achieve that：
training_accuracy = accuracy_score(y_trainingSet_quantizing, best_y_train_pred)
print(f"\033[1mQ2(c)The Training Accuracy is: {training_accuracy:.4f}\033[0m")

test_accuracy = accuracy_score(y_testSet_quantizing, best_y_test_pred)
print(f"\033[1mQ2(c)The Test Accuracy is: {test_accuracy:.4f}\033[0m")

# (d)
param_grid = {'C': C_100_values}
grid_lr = GridSearchCV(estimator=LogisticRegression(penalty='l2', solver='lbfgs'),cv=5, param_grid=param_grid)
grid_lr.fit(X_trainingSet, y_trainingSet_quantizing)

# (d)
# get the best C using gridsearch. We can find that the C value is different from (c)-37.649358067924716.
best_C = grid_lr.best_params_['C']
print(f"\033[1mQ2(d):\033[0m")
print(f"\033[1mQ2(d) Default Best C value: {best_C}\033[0m")

# (d)
# Re-run the code with some changes to give consistent results to those we computed by hand.

# Here we use shuffle = False to achieve the same effect as manually split the X trainingSet into 5 folds based on the original data index.
cv_close_shuffle = KFold(n_splits=5, shuffle=False)

# 'neg_log_loss' is a standard scoring metric built in GridSearchCV, the negative is because in the Scoring criteria, the larger value means the better choice, but log-loss is the samller the best,
# so we use -log-loss 'neg_log_loss' here to fit the scoring principle.
grid_lr = GridSearchCV(estimator=LogisticRegression(penalty='l2', solver='lbfgs'), cv=cv_close_shuffle, param_grid=param_grid, scoring='neg_log_loss')
grid_lr.fit(X_trainingSet, y_trainingSet_quantizing)

# Best C value after rerun the GridSearchCV:
best_C_rerun = grid_lr.best_params_['C']
print(f"\033[1mQ2(d) After Re-run Best C value: {best_C_rerun}\033[0m")
best_log_loss = -grid_lr.best_score_
print(f"\033[1mQ2(d) After Re-run Best log-loss value: {-grid_lr.best_score_}\033[0m")
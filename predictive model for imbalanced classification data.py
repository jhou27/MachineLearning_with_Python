#import necessary packages
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from collections import Counter
from numpy import mean
from numpy import std

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

###### explore the dataset ###################################################
# define the dataset location
filename = 'phoneme.csv'
# load the csv file as a data frame
df = read_csv(filename, header=None)
# define a mapping of class values to colors
color_dict = {0:'blue', 1:'red'}
# map each row to a color based on the class value
colors = [color_dict[x] for x in df.values[:, -1]]
# drop the target variable
inputs = DataFrame(df.values[:, :-1])
# create pairwise scatter plots of numeric input variables
scatter_matrix(inputs, diagonal='kde', color=colors)
pyplot.show()
# create histograms of numeric input variables
df.hist()
pyplot.show()


###### baseline model #########################################

#baseline model evaluation
# load the dataset
def load_dataset(full_path):
 # load the dataset as a numpy array
 data = read_csv(full_path, header=None)
 # retrieve numpy array
 data = data.values
 # split into input and output elements
 X, y = data[:, :-1], data[:, -1]
 return X, y
 
# evaluate a model
def evaluate_model(X, y, model):
 # define evaluation procedure
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # define the model evaluation the metric
 metric = make_scorer(geometric_mean_score)
 # evaluate model
 scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
 return scores
 
# define the location of the dataset
full_path = 'phoneme.csv'
# load the dataset
X, y = load_dataset(full_path)
# summarize the loaded dataset
print(X.shape, y.shape, Counter(y))
# define the reference model
model = DummyClassifier(strategy='uniform')
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
print('Mean G-Mean: %.3f (%.3f)' % (mean(scores), std(scores)))

###### test different classfication model #########################################
# We will evaluate the following machine learning models:

# Logistic Regression (LR)
# Support Vector Machine (SVM)
# Bagged Decision Trees (BAG)
# Random Forest (RF)
# Extra Trees (ET)

# define models to test
def get_models():
 models, names = list(), list()
 # LR
 models.append(LogisticRegression(solver='lbfgs'))
 names.append('LR')
 # SVM
 models.append(SVC(gamma='scale'))
 names.append('SVM')
 # Bagging
 models.append(BaggingClassifier(n_estimators=1000))
 names.append('BAG')
 # RF
 models.append(RandomForestClassifier(n_estimators=1000))
 names.append('RF')
 # ET
 models.append(ExtraTreesClassifier(n_estimators=1000))
 names.append('ET')
 return models, names
 
# define the location of the dataset
full_path = 'phoneme.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
 # evaluate the model and store results
 scores = evaluate_model(X, y, models[i])
 results.append(scores)
 # summarize and store
 print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


###### evaluate data oversampling algos #########################################
# We will test five different oversampling methods; specifically:

# Random Oversampling (ROS)
# SMOTE (SMOTE)
# BorderLine SMOTE (BLSMOTE)
# SVM SMOTE (SVMSMOTE)
# ADASYN (ADASYN)


# define oversampling models to test
def get_models():
 models, names = list(), list()
 # RandomOverSampler
 models.append(RandomOverSampler())
 names.append('ROS')
 # SMOTE
 models.append(SMOTE())
 names.append('SMOTE')
 # BorderlineSMOTE
 models.append(BorderlineSMOTE())
 names.append('BLSMOTE')
 # SVMSMOTE
 models.append(SVMSMOTE())
 names.append('SVMSMOTE')
 # ADASYN
 models.append(ADASYN())
 names.append('ADASYN')
 return models, names
 
# define the location of the dataset
full_path = 'phoneme.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
 # define the model
 model = ExtraTreesClassifier(n_estimators=1000)
 # define the pipeline steps
 steps = [('s', MinMaxScaler()), ('o', models[i]), ('m', model)]
 # define the pipeline
 pipeline = Pipeline(steps=steps)
 # evaluate the model and store results
 scores = evaluate_model(X, y, pipeline)
 results.append(scores)
 # summarize and store
 print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
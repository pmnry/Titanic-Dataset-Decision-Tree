import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Kaggle Titanic Dataset
# In this notebook we test the implementation of a simple Decision Tree (QUEST algorithm) on the Titanic Dataset from Kaggle.
# We will first run some data analysis and data visualization and then implement the model and analyze its results.

df_train = pd.read_csv('train.csv')
# df_train.dropna(inplace=True)

df_test = pd.read_csv('test.csv')
# df_test.dropna(inplace=True)

NumImp = SimpleImputer(missing_values=np.nan, strategy='median')
NumCol = ['Age', 'Fare']

CatImp = SimpleImputer(strategy='constant')
CatCol = ['Embarked', 'Sex', 'Pclass']

for col in NumCol:
    df_train[col] = NumImp.fit_transform(df_train[col].values.reshape(-1,1))
    df_test[col] = NumImp.transform(df_test[col].values.reshape(-1, 1))

for col in CatCol:
    df_train[col] = CatImp.fit_transform(df_train[col].values.reshape(-1,1))
    df_test[col] = CatImp.transform(df_test[col].values.reshape(-1,1))

### Print columns types in preparation of our future steps

print(df_train.dtypes)


### Find out more about the data structure

df_train.describe()


# About 40% of the passengers survived. The mean age was 29 years with the youngest passenger not older than 6 months.
# The fare seems to range between rather extreme values. This could be a discriminant factor when trying to predict the survival of a passenger.

### Fare distribution among the population

sns.distplot(df_train['Fare'], kde=False, rug=False)
plt.show()

### Is Fare linked to a higher survival rate?

sns.boxplot(x="Survived", y="Fare", data=df_train)
plt.show()

### Creating discrete variables out of categorical data
# If we want to be able to use variables such as 'Embarked' in our classifier we need to modify these in order to turn them into discrete variable. To do that we use the `LabelEncoder` which is part of the `sklearn` package.

categorical_feats = ['Embarked', 'Sex']

for feat in categorical_feats:
    le = preprocessing.LabelEncoder()
    le.fit(df_train[feat].unique())
    df_train[feat] = le.transform(df_train[feat])
    df_test[feat] = le.transform(df_test[feat])

### Implementing a PCA decomposition

features = ['Fare', 'Embarked', 'Sex', 'Age']
pca = PCA(n_components=4)
pca.fit(df_train[features])
print(pca.explained_variance_ratio_)

### Fitting a simple Decision Tree

features = ['Fare', 'Embarked', 'Sex', 'Age']
model = DecisionTreeClassifier()
fit_model = model.fit(df_train[features], df_train['Survived'])
y_pred = fit_model.predict(df_train[features])


### Displaying its accuracy

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(df_train['Survived'], y_pred))

### Testing Random Forest with hyper parameters tuning

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


### Fitting a simple Random Forest

features = ['Fare', 'Embarked', 'Sex', 'Age']
model_RF = RandomForestClassifier()
model_RF_random = RandomizedSearchCV(estimator=model_RF, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                     random_state=42, n_jobs=-1)
model_RF_random.fit(df_train[features], df_train['Survived'])
y_pred_RF = model_RF_random.predict(df_train[features])

### Displaying its accuracy

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(df_train['Survived'], y_pred_RF))

### Prediction on test set

y_pred_RF_test = model_RF_random.predict(df_test[features])
df_results = pd.DataFrame(y_pred_RF_test, index=df_test['PassengerId'], columns=['Survived'])
df_results.to_csv('results.csv')
#!/usr/bin/env python
# coding: utf-8

# ## Importing Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn import metrics


# ## Data Preprocessing

# In[2]:


pd.set_option('display.max_columns',90)
data=pd.read_csv(r'C:\Users\ASUS\Downloads\Life Expectancy Data.csv')
data


# In[3]:


data.describe(include='all')


# In[4]:


data.isnull().sum()


# In[5]:


data.columns = data.columns.str.strip()


# In[6]:


data.dtypes


# In[7]:


data['Adult Mortality']=data['Adult Mortality'].fillna(value=data['Adult Mortality'].mean())
data['Alcohol']=data['Alcohol'].fillna(value=data['Alcohol'].mean())
data['Hepatitis B']=data['Hepatitis B'].fillna(value=data['Hepatitis B'].mean())
data['BMI']=data['BMI'].fillna(value=data['BMI'].mean())
data['Polio']=data['Polio'].fillna(value=data['Polio'].mean())
data['Total expenditure']=data['Total expenditure'].fillna(value=data['Total expenditure'].mean())
data['Diphtheria']=data['Diphtheria'].fillna(value=data['Diphtheria'].mean())
data['GDP']=data['GDP'].fillna(value=data['GDP'].mean())
data['Population']=data['Population'].fillna(value=data['Population'].mean())
data['thinness  1-19 years']=data['thinness  1-19 years'].fillna(value=data['thinness  1-19 years'].mean())
data['thinness 5-9 years']=data['thinness 5-9 years'].fillna(value=data['thinness 5-9 years'].mean())
data['Income composition of resources']=data['Income composition of resources'].fillna(value=data['Income composition of resources'].mean())
data['Schooling']=data['Schooling'].fillna(value=data['Schooling'].mean())
data['Life expectancy']=data['Life expectancy'].fillna(value=data['Life expectancy'].mean())


# In[8]:


data.isnull().sum()


# In[9]:


data.corr()['Life expectancy']


# In[10]:


avarage_corr=abs(data.corr()['Life expectancy']).mean()


# In[11]:


avarage_corr


# In[12]:


data.columns


# In[13]:


dropped_columns = []

for i in data[['Year', 'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI',
       'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria',
       'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years',
       'thinness 5-9 years', 'Income composition of resources', 'Schooling',
       'Life expectancy']]:
    
    if abs(data.corr()['Life expectancy'][i])<avarage_corr:
        dropped_columns.append(i)
    
data.drop(dropped_columns, axis=1, inplace=True)  


# In[14]:


data


# In[15]:


data.columns


# In[16]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data [[
    'Adult Mortality', 
    'BMI', 
#     'Polio', 
#     'Diphtheria',  
    'HIV/AIDS', 
    'GDP', 
    'thinness  1-19 years', 
#     'thinness 5-9 years',
#     'Income composition of resources', 
#     'Schooling'
]]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif


# In[17]:


data.drop(['Polio', 'Diphtheria', 'thinness 5-9 years', 'Income composition of resources', 'Schooling'], axis=1, inplace=True)


# In[18]:


data.columns


# In[19]:


for i in data[['Adult Mortality', 'BMI', 'HIV/AIDS', 'GDP',
       'thinness  1-19 years', 'Life expectancy']]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[20]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[21]:


for i in data[['Adult Mortality', 'BMI', 'HIV/AIDS', 'GDP',
       'thinness  1-19 years', 'Life expectancy']]:
    
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])


# In[22]:


for i in data[['Adult Mortality', 'BMI', 'HIV/AIDS', 'GDP',
       'thinness  1-19 years', 'Life expectancy']]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[23]:


data=data.reset_index(drop=True)


# In[24]:


data.describe(include='all')


# In[25]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(10,5))

ax1.scatter(data['Adult Mortality'],data['Life expectancy'])
ax1.set_title('Adult Mortality and Life expectancy')
ax1.set_xlabel('Adult Mortality')
ax1.set_ylabel('Life expectancy')
ax2.scatter(data['BMI'],data['Life expectancy'])
ax2.set_title('BMI and Life expectancy')
ax2.set_xlabel('BMI')
ax2.set_ylabel('Life expectancy')

plt.show()


# In[26]:


log_life_expectancy = np.log(data['Life expectancy'])
data['Log_life_expectancy']=log_life_expectancy


# In[27]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(10,5))

ax1.scatter(data['Adult Mortality'],data['Log_life_expectancy'])
ax1.set_title('Adult Mortality and Life expectancy')
ax1.set_xlabel('Adult Mortality')
ax1.set_ylabel('Life expectancy')
ax2.scatter(data['BMI'],data['Log_life_expectancy'])
ax2.set_title('BMI and Life expectancy')
ax2.set_xlabel('BMI')
ax2.set_ylabel('Life expectancy')

plt.show()


# In[28]:


data


# In[29]:


data_with_dummies=data.drop('Country', axis=1)


# In[30]:


data_with_dummies=pd.get_dummies(data_with_dummies, drop_first=True)


# In[31]:


data_with_dummies


# In[32]:


data_with_dummies.columns


# In[33]:


data_with_dummies=data_with_dummies[['Status_Developing', 'Adult Mortality', 'BMI', 'HIV/AIDS', 'GDP', 'thinness  1-19 years', 'Log_life_expectancy']]


# In[34]:


data_with_dummies


# In[35]:


sc=StandardScaler()


# In[36]:


scaled_data=sc.fit_transform(data_with_dummies)


# In[37]:


scaled_data


# In[38]:


scaled_data=pd.DataFrame(scaled_data, columns=data_with_dummies.columns)
scaled_data


# ## Modeling

# In[39]:


X=scaled_data.drop('Log_life_expectancy', axis=1)
y=scaled_data['Log_life_expectancy']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


def evaluate(model, X_test, y_test):
    
    y_pred_test=model.predict(X_test)
    mae_test=metrics.mean_absolute_error(y_test, y_pred_test)
    mse_test=metrics.mean_squared_error(y_test, y_pred_test)
    rmse_test=np.sqrt(mse_test)
    r2_test=metrics.r2_score(y_test, y_pred_test)
    
    
    y_pred_train=model.predict(X_train)
    mae_train=metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train=metrics.mean_squared_error(y_train, y_pred_train)
    rmse_train=np.sqrt(mse_train)
    r2_train=metrics.r2_score(y_train, y_pred_train)
    
    
    results_dict = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Train': [mae_train, mse_train, rmse_train, r2_train*100],
        'Test': [mae_test, mse_test, rmse_test, r2_test*100]
    }

    results_df = pd.DataFrame(results_dict)
    
    print(results_df)


# In[42]:


lr=LinearRegression()
lr.fit(X_train, y_train)


# In[43]:


result_lr=evaluate(lr, X_test, y_test)


# ## Modeling for other ML algorithms

# In[44]:


data


# In[45]:


data.drop('Log_life_expectancy', axis=1, inplace=True)


# In[46]:


data_dummied=data.drop('Country', axis=1)


# In[47]:


data_dummied=pd.get_dummies(data_dummied, drop_first=True)


# In[48]:


data_dummied


# In[49]:


data_dummied.columns


# In[50]:


data_dummied=data_dummied[['Status_Developing', 'Adult Mortality', 'BMI', 'HIV/AIDS', 'GDP', 'thinness  1-19 years',
       'Life expectancy']]


# In[51]:


data_dummied


# In[52]:


X=data_dummied.drop('Life expectancy', axis=1)
y=data_dummied['Life expectancy']


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[54]:


def evaluate(model, X_test, y_test):
    y_pred=model.predict(X_test)
    mae_test=metrics.mean_absolute_error(y_test, y_pred)
    mse_test=metrics.mean_squared_error(y_test, y_pred)
    rmse_test=np.sqrt(mse_test)
    r2_test=metrics.r2_score(y_test, y_pred)
    
    y_pred_train=model.predict(X_train)
    mae_train=metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train=metrics.mean_squared_error(y_train, y_pred_train)
    rmse_train=np.sqrt(mse_train)
    r2_train=metrics.r2_score(y_train, y_pred_train)
    
    
    results_dict = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Train': [mae_train, mse_train, rmse_train, r2_train*100],
        'Test': [mae_test, mse_test, rmse_test, r2_test*100]
    }

    results_df = pd.DataFrame(results_dict)
    
    print(results_df)


# In[55]:


dtr=DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# In[56]:


result_dtr=evaluate(dtr, X_test, y_test)


# In[57]:


base_rfr=RandomForestRegressor(n_estimators=100, random_state=42)
base_rfr.fit(X_train, y_train)


# In[58]:


result_base_rfr=evaluate(base_rfr, X_test, y_test)


# In[59]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[60]:


rfr_randomized = RandomizedSearchCV(estimator = base_rfr, 
                                    param_distributions = random_grid, 
                                    n_iter = 10, cv = 3, 
                                    verbose=1, random_state=42, 
                                    n_jobs = -1)

rfr_randomized.fit(X_train, y_train)


# In[61]:


rfr_randomized.best_params_


# In[62]:


optimized_rfr_model=rfr_randomized.best_estimator_
result_optimized_rfr_model=evaluate(optimized_rfr_model, X_test, y_test)


# In[63]:


xgboost_reg_base=XGBRegressor()
xgboost_reg_base.fit(X_train, y_train)


# In[64]:


result_xgboost_reg_base=evaluate(xgboost_reg_base, X_test, y_test)


# In[65]:


lightgbm_reg_base=LGBMRegressor()
lightgbm_reg_base.fit(X_train, y_train)


# In[66]:


result_lightgbm_reg_base=evaluate(lightgbm_reg_base, X_test, y_test)


# In[67]:


catboost_dummy_base=CatBoostRegressor()
catboost_dummy_base.fit(X_train, y_train)


# In[68]:


result_catboost_dummy_base=evaluate(catboost_dummy_base, X_test, y_test)


# In[69]:


#Hyperparameter Tuning (Lightgbm)
param_distributions = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]    
}

param_distributions


# In[70]:


lightgbm_randomized=RandomizedSearchCV(lightgbm_reg_base, param_distributions=param_distributions, 
                                       n_iter=10, cv=5, 
                                       n_jobs=-1, random_state=42)

lightgbm_randomized.fit(X_train, y_train)


# In[71]:


print('Best hyperparameters for the lightgbm:', lightgbm_randomized.best_params_)


# In[72]:


optimized_lightgbm=lightgbm_randomized.best_estimator_


# In[73]:


result_lightgbm_optimized = evaluate(optimized_lightgbm, X_test, y_test)


# In[74]:


#Hyperparameter Tuning (XGBoost)
param_distributions = {
    
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'subsample': np.linspace(0.5, 1, num=6),
    'colsample_bytree': np.linspace(0.5, 1, num=6),
    'gamma': [0,1,5,10]
    
}

param_distributions


# In[75]:


xgboost_randomized = RandomizedSearchCV(xgboost_reg_base, param_distributions=param_distributions,
                                        n_iter=10, cv=5,
                                        n_jobs=-1, random_state=42)

xgboost_randomized.fit(X_train, y_train)


# In[76]:


print('Best hyperparameters for XGBoost:', xgboost_randomized.best_params_)


# In[77]:


optimized_xgboost=xgboost_randomized.best_estimator_


# In[78]:


result_xgboost_optimized= evaluate(optimized_xgboost, X_test, y_test)


# In[79]:


#Hyperparameter Tuning (CatBoost)

param_distributions = {
    
    'iterations': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': np.linspace(2, 30, num=7)
    
}

param_distributions


# In[80]:


catboost_randomized=RandomizedSearchCV(catboost_dummy_base,
                                       param_distributions=param_distributions,
                                       cv=5, n_iter=10, 
                                       random_state=42)

catboost_randomized.fit(X_train, y_train)


# In[81]:


print('Best parameters for CatBoost model:', catboost_randomized.best_params_)


# In[82]:


optimized_catboost=catboost_randomized.best_estimator_


# In[83]:


result_catboost_optimized = evaluate(optimized_catboost, X_test, y_test)


# In[84]:


#Stacking Model

base_regressors = [
    optimized_rfr_model,
    xgboost_reg_base,
    lightgbm_reg_base,
    optimized_catboost,
]


# In[85]:


meta_regressor=catboost_dummy_base


# In[86]:


stacking_regressor = StackingCVRegressor(regressors=base_regressors,
                                           meta_regressor=meta_regressor,
                                           cv=5,
                                           use_features_in_secondary=True,
                                           verbose=1,
                                           random_state=42)


# In[87]:


stacking_regressor.fit(X_train, y_train)


# In[88]:


result_stacking_regressor=evaluate(stacking_regressor, X_test, y_test)


# In[89]:


svr_base_model=SVR()
svr_base_model.fit(X_train, y_train)


# In[90]:


result_svr_base_model=evaluate(svr_base_model, X_test, y_test)


# In[91]:


kernel = ['linear', 'poly', 'rbf', 'sigmoid']

gamma = ['scale', 'auto'] 

C = [1, 10, 100, 1e3, 1e4]

epsilon = [0.1 , 0.01, 0.001]



random_grid = {'kernel': kernel,
               'gamma': gamma,
               'C': C,
               'epsilon': epsilon}
print(random_grid)


# In[92]:


svr_randomized = RandomizedSearchCV(estimator =svr_base_model, 
                                    param_distributions = random_grid, 
                                    n_iter = 1, cv = 2, 
                                    verbose=1, 
                                    n_jobs = -1)

svr_randomized.fit(X_train, y_train)


# In[93]:


svr_randomized.best_params_


# In[94]:


svr_optimized=svr_randomized.best_estimator_


# In[95]:


result_svr_optimized=evaluate(svr_optimized, X_test, y_test)


# In[96]:


optimized_svr_model_2= SVR(kernel='rbf', gamma='scale', epsilon=0.001, C=1000)
optimized_svr_model_2.fit(X_train, y_train) 
result_optimized_svr_model_2=evaluate(optimized_svr_model_2, X_test, y_test)


# ## Univariate Analysis
# 

# In[97]:


variables = []
train_r2_scores = []
test_r2_scores = []

for i in X_train.columns: 
    X_train_single = X_train[[i]]
    X_test_single = X_test[[i]]

    
    catboost_dummy_base.fit(X_train_single, y_train)
    
    
    y_pred_train_single = catboost_dummy_base.predict(X_train_single)
    train_r2 = metrics.r2_score(y_train, y_pred_train_single)
    
    

    y_pred_test_single = catboost_dummy_base.predict(X_test_single)
    test_r2 = metrics.r2_score(y_test, y_pred_test_single)

    variables.append(i)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    
    
    
results_df = pd.DataFrame({'Variable': variables, 'Train R2': train_r2_scores, 'Test R2': test_r2_scores})

results_df_sorted = results_df.sort_values(by='Test R2', ascending=False)

pd.options.display.float_format = '{:.4f}'.format

results_df_sorted


# ## Catboost model with categorical columns

# In[98]:


data


# In[99]:


X=data.drop('Life expectancy', axis=1)
y=data['Life expectancy']


# In[100]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)


# In[101]:


def evaluate(model, X_test, y_test):
    y_pred=model.predict(X_test)
    mae_test=metrics.mean_absolute_error(y_test, y_pred)
    mse_test=metrics.mean_squared_error(y_test, y_pred)
    rmse_test=np.sqrt(mse_test)
    r2_test=metrics.r2_score(y_test, y_pred)
    
    y_pred_train=model.predict(X_train)
    mae_train=metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train=metrics.mean_squared_error(y_train, y_pred_train)
    rmse_train=np.sqrt(mse_train)
    r2_train=metrics.r2_score(y_train, y_pred_train)
    
    
    results_dict = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Train': [mae_train, mse_train, rmse_train, r2_train*100],
        'Test': [mae_test, mse_test, rmse_test, r2_test*100]
    }

    results_df = pd.DataFrame(results_dict)
    
    print(results_df)


# In[102]:


catboost_cat_model=CatBoostRegressor(cat_features=['Country', 'Status'])
catboost_cat_model.fit(X_train, y_train)


# In[103]:


result_catboost_cat_model=evaluate(catboost_cat_model, X_test, y_test)


# In[106]:


from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

categorical_cols = ['Country', 'Status']

variables = []
train_r2_scores = []
test_r2_scores = []

catboost_regressor = CatBoostRegressor()

for col in X.columns:
 
    X_train_single = X_train[[col]]
    X_test_single = X_test[[col]]

    cat_features_indices = [X_train_single.columns.get_loc(c) for c in categorical_cols if c in X_train_single]
    train_pool = Pool(X_train_single, label=y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test_single, label=y_test, cat_features=cat_features_indices)
    
   
    catboost_regressor.fit(train_pool, verbose=False)
    
    
    y_pred_train = catboost_regressor.predict(train_pool)
    train_r2 = r2_score(y_train, y_pred_train)
    
    
    y_pred_test = catboost_regressor.predict(test_pool)
    test_r2 = r2_score(y_test, y_pred_test)

    
    variables.append(col)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)


results_df = pd.DataFrame({'Variable': variables, 'Train R2': train_r2_scores, 'Test R2': test_r2_scores})

results_df_sorted = results_df.sort_values(by='Test R2', ascending=False)


results_df_sorted


# In[ ]:





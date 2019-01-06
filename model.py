import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.utils import resample
from sklearn import model_selection
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#Loading the train and test data
train_data=pd.read_csv("Yes_Bank_Training.csv")
test_data=pd.read_csv("Yes_Bank_Test.csv")

# Dataset info
print(train_data.shape)
print(train_data.info())
print(train_data.columns)
print(test_data.shape)
print(test_data.columns)

# Visualizing imbalanced train_dataset
sns.countplot("outcome",data=train_data)
plt.show()

########## Data Preprocessing ##########

#checking the obj type attributes
train_data.columns[train_data.dtypes.eq(object)]
enc=LabelEncoder()
#Train data attributes categorical to numeric
train_data['job_description']=enc.fit_transform(train_data['job_description'])
train_data['marital_status']=enc.fit_transform(train_data['marital_status'])
train_data['education_details']=enc.fit_transform(train_data['education_details'])
train_data['has_default']=enc.fit_transform(train_data['has_default'])
train_data['housing_status']=enc.fit_transform(train_data['housing_status'])
train_data['previous_loan']=enc.fit_transform(train_data['previous_loan'])
train_data['phone_type']=enc.fit_transform(train_data['phone_type'])
train_data['month_of_year']=enc.fit_transform(train_data['month_of_year'])
train_data['poutcome_of_campaign']=enc.fit_transform(train_data['poutcome_of_campaign'])
train_data['outcome']=enc.fit_transform(train_data['outcome'])

#Test data attributes categorical to numeric
test_data['job_description']=enc.fit_transform(test_data['job_description'])
test_data['marital_status']=enc.fit_transform(test_data['marital_status'])
test_data['education_details']=enc.fit_transform(test_data['education_details'])
test_data['has_default']=enc.fit_transform(test_data['has_default'])
test_data['housing_status']=enc.fit_transform(test_data['housing_status'])
test_data['previous_loan']=enc.fit_transform(test_data['previous_loan'])
test_data['phone_type']=enc.fit_transform(test_data['phone_type'])
test_data['month_of_year']=enc.fit_transform(test_data['month_of_year'])
test_data['poutcome_of_campaign']=enc.fit_transform(test_data['poutcome_of_campaign'])

df=train_data
df_majority=df[df['outcome']==0]
print(df_majority.shape)
df_minority=df[df['outcome']==1]
print(df_minority.shape)

# # DownSampling Majority class (Under_Sampling)
# df_majority_new=resample(df_majority,replace=False,n_samples=1840,random_state=123)
# train_data_new=pd.concat([df_majority_new,df_minority])
# print(train_data_new.shape)

# UpSampling Minority class (Over_Sampling)
# df_minority_new=resample(df_minority,replace=True,n_samples=29809,random_state=123)
# train_data_new=pd.concat([df_majority,df_minority_new])


# Preparing train and test data
x_train=train_data.drop(['outcome'],1)
y_train=train_data['outcome']
x_test=test_data

# Training the classifier
clf=XGBClassifier(n_estimators=1000,max_depth=6,learning_rate=0.1,subsample=0.8,col_sample_bytree=0.8,objective='binary:logistic',scale_pos_weight=1,eval_metric='auc',iid=False,cv=5)
clf.fit(x_train,y_train)

# Prediction
y_pred=clf.predict(x_test)
y_pred=pd.DataFrame(y_pred)

plot_importance(clf)
plt.figure(figsize=(30,30))
plt.show()

serial_number=test_data.iloc[:,0]
result=pd.concat([serial_number,y_pred],axis=1)
d={0:'no',1:'yes'}
result[0]=result[0].map(d)
result.to_csv("result.csv",index=False)




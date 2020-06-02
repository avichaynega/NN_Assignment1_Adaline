import pandas as pd  
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

from adalinegd import AdalineGD
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix


def build_and_evaluate_model(x, y):
	
	#use sklearn library to split the dataset : 33% to test and 66% to train
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,shuffle = True, random_state=None)
	
	#change the values of desire output both y_train and y_test 
	#N- nonrecurre= -1 ,R - recurre =1 
	y_train = np.where(y_train=='N',-1,1)
	y_test =  np.where(y_test=='N',-1,1)
	
	start = timeit.default_timer()
	#train Adaline

	model1 = AdalineGD(n_iter = 2000, eta = 1e-9)
	model1.fit(x_train, y_train)
	y_train_pred = model1.predict(x_train)

	print('************train ************')
	print('Misclassified samples: %d' % (y_train != y_train_pred).sum())
	accuracy_train = accuracy_score(y_train, y_train_pred) * 100
	print('Accuracy:  %.2f' % accuracy_train,'%')

	#test Adaline 
	print('************test ************')
	y_pred = model1.predict(x_test)

	stop = timeit.default_timer()
	execution_time = stop - start

	#time takes to train and test  in seconds
	print('Program Executed in : %.2f' % execution_time , 'sec') 

	print('Misclassified samples: %d' % (y_test != y_pred).sum())
	accuracy = accuracy_score(y_test, y_pred) * 100
	print('Accuracy:  %.2f' % accuracy,'%')
	
	# plot the confusion matrix 
	cm = confusion_matrix(y_test, y_pred)
	sns.heatmap(cm,annot=True,fmt="d")
	plt.show()
	return accuracy
	
	


#this array represent the features name of dataset 
column_names = ['ID','Outcome','Time', 'Radius Mean', 'Texture Mean', 'Perimeter Mean',
	   'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean',
	   'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
	   'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE',
	   'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE',
	   'Fractal Dimension SE', 'Radius Worst', 'Texture Worst',
	   'Perimeter Worst', 'Area Worst', 'Smoothness Worst',
	   'Compactness Worst', 'Concavity Worst', 'Concave Points Worst',
	   'Symmetry Worst', 'Fractal Dimension Worst', 'Tumor Size', 'Lymph Node Status']


raw_dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data', names=column_names,
					  na_values = "?", sep=",")

dataset = raw_dataset.copy()

#clean data from 'nan' values rows , in this dataset the 'nan' values appears as '?'
dataset = dataset.dropna()
y = dataset['Outcome']

#drop irrelevant columns
drop_list = ['ID', 'Time', 'Outcome']
X = dataset.drop(drop_list, 1)

#plot the dataset cut-off  to Recurr and Nonrecurr
# ax = sns.countplot(y,label="Count")
# plt.show()
# N, R = y.value_counts()
# print('Number of Recurring Cases: ', R)
# print('Number of Non-Recurring Cases: ',N)

#run the function 3 time to see the avarage and standart deviation of results 
#Note that every time the function has been called ,the split of values change
result1 = build_and_evaluate_model(X, y)


result2 = build_and_evaluate_model(X, y)


result3 = build_and_evaluate_model(X, y)	


results = np.array([result1,result2,result3])

print ('Average : %.2f' % np.average(results) )
print('Standard deviation : %.2f' % np.std(results))

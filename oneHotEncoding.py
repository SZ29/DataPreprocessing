'''Categorical data is not accepted by some one the machine learning and deep learning algorithms
However we can make it work by converting this type of data into numerical data to benefit from the
power of those classifiers.
All categorical variables are called nominal variables the ones which have some kind of natural sequence are called
ordinal variables

Example:
1- Monday, Tuesday, Wednesday, Thursday, Friday they are ordinal variables and can be represented by 1,2,3,4,5
as number of weekday.
2 Sunny, Cloudy, Rainy are nominal but since they donot have any natural relationship they are not ordinal.

There are two types of Encoding used while the preprocessing step

1- Label Encoding/Integar Encoding : Each category is 'labelled' or assigned an integar eg: Monday = 1, Tuesday =2,
Wednesday = 3, Thursday = 4, Friday = 5, Saturday = 6, Sunday = 7 .
Label encoding helps machine learning algorithm to identify that there exist a natural sequence in the data and they can use it.
2- One hot enncoding: This is done when there is no natural sequence in the categories eg: colors in case of red, yellow,
blue we will need 3 binary variables
    R   Y   B
1-  1   0   0
2-  0   1   0
3-  0   0   1

 Hot encoding allows more expression machine learning algorithms that cannot deal with categorical data
Here we are doing both encoding using ScikitLearn
'''
from sklearn import preprocessing
#Label Encoding

#Dataset of 10 entries(training)
train_data = ['autumn','autumn','summer','spring','winter','summer','spring','spring','autumn','summer']

#initializing label Encoder
Label_Encoder = preprocessing.LabelEncoder()

#fitting data to the label encoder, ie assigning values
model = Label_Encoder.fit(train_data)

#number of distinct classes = 4 ['autumn' 'spring' 'summer' 'winter']
print(model.classes_)

#Test_data
test_data = ['summer','winter','winter','autumn','winter']
test_labels = Label_Encoder.transform(test_data)

#In printing we get [2 3 3 0 3]
print(test_labels)

#reversing these and checking if we get our test set back
test_labelsReverse = Label_Encoder.inverse_transform(test_labels)

#Checking if we get data equal to out test data.
print(test_labelsReverse)

#OneHot Encoding

print("#####################ONE HOT ENCODING EXAMPLE######################\n")
OneHot_Encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
train_data1 = [['Blue', 1], ['Red', 3], ['Green', 2],['Green',4],['Red',5]]
OneHot_Encoder.fit(train_data1)
categories = OneHot_Encoder.categories_
print(categories)

test_data1 = [['Green', 1], ['Red', 4]]
print('Running one-hot encoder on test data:')
test_labels1 =OneHot_Encoder.transform(test_data1)
print(test_labels1.toarray())
test_labelsReverse1 = OneHot_Encoder.inverse_transform(test_labels1)
print('Checking if labels are correct:')
print(test_labelsReverse1)
features = OneHot_Encoder.get_feature_names()
print('Features')
print(features)

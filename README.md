DataPreprocessing

This repository contains different pre-processing techniques used to tune your data according to the machine learning algorithm you use.

HOT ENCODING AND LABEL ENCODING

Categorical data is not accepted by some one the machine learning and deep learning algorithms
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
Check out Encoding.py for code.

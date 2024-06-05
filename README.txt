README!!!

This is the code to create an Naive Bayes Classifier (NBC) used for Sentiment Analysis

The training dataset is provided inside the "data" folder with the file name of "train.csv"
The testing dataset is also provided inside the "data" folder with the file name of "test.csv"

There are 5 files to create and predict from a trained NBC model:
	1. main.py
	2. plotting.py
	3. predict.py
	4. preprocess.py
	5. training.py

The function of each files is as follow:
	1. main.py: the driver code to run all the NBC for Sentiment Analysis
	2. plotting.py: to plot the accuracy of the NBC model
	3. predict.py: to predict the unknown dataset (test dataset) using a trained NBC model
	4. preprocess.py: to pre-process the text by:
		a. Lowercasing all of the text
		b. Remove special characters: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
		c. Remove stopwords (the list is inside the "stopwords.txt" file)
	5. training.py: to train the NBC model using the training dataset


Instructions (windows):
1. Make sure the 5 files are in the same folder
2. If needed, change the test and train dataset file path
3. Run "main.py" file!


Dependencies/Requirements:
1. matplotlib >= 3.5.1

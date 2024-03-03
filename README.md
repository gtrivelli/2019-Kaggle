# Abstract

We participated in Kaggle’s 2019 Data Science Bowl taking data from a children’s educational application. In the data, game sessions are labeled how likely it is for the installation id in that game session to pass the test within a certain number of tries. We attempted to predict what label would be assigned to installation ids given the events for each game session. We could unfortunately not overcome the issues surrounding the data and instead of obtaining a solution for the problem, found insight in to real world data mining. 

# Introduction

## Background
	
Kaggle hosts a yearly data science competition called the Data Science Bowl. It is the world’s largest data science competition that focuses on social good. From heart disease detection to ocean health, the competition provides an opportunity for data scientists to have a positive impact in the world.

This year, the 2019 Data Science Bowl aims to aid in early childhood development using PBS KIDS Measure Up! app. The app is a game-based learning tool developed as a part of the CPB-PBS Ready To Learn Initiative with funding from the U.S. Department of Education. It teaches early STEM concepts like length, width, capacity and weight in an adventure format with their favorite PBS KIDS characters. Using the in-app assessment score and children's path through the game, we will be predicting scores on in-game assessments.

## Problem Definition

We are provided with game analytics for the PBS KIDS Measure Up! app. In this app, children navigate a map and complete various levels, which may be activities, video clips, games, or assessments that test their comprehension skills. The intent of the competition is to use the gameplay data to forecast how many attempts a child will take to pass a given assessment. Both correct and incorrect answers are considered attempts. Each instance of the application’s installation has an id, so it does not necessarily represent a single child. Each id is assigned a value from 0 to 3:

3: the assessment was solved on the first attempt  
2: the assessment was solved on the second attempt  
1: the assessment was solved after 3 or more attempts  
0: the assessment was never solved  

Given data for different installations, we must predict what value the installation would have for each assessment. 

## Motivation

Initially, we were going to make use of some data on the University of California, Irvine database concerning Victorian Era authorship attribution for our project. Ultimately, we decided against this direction for two major reasons. 

First, we ran into a potential issue with the testing set where some authors were unidentified by the training set. In order to specifically predict these unknown authors, we would have had to gather an unknown amount of additional data to improve our training set. While doable, this process would have significantly increased the difficulty of the project. Additionally, this would require spending a significant amount of time on gathering data, which is not the focus of the course. Alternatively, we could have ignored the task of identifying these unknown authors. Had we gone that route, however, the difficulty of the project may not have been sufficient to achieve a satisfactory grade.

We decided to take some time and let the dilemma simmer on the back burner. During this time, the weekend after our initial pitch, Professor Lin posted a link to the Data Science Bowl problem to Piazza, and suggested to the class that participation in the Bowl could feasibly constitute a final project. This idea appealed to us for two reasons. The first is the potential to win prize money. The second being that this is a real world application of data mining that could be included on a resume. 


# Solution

## Resources

https://measureup.pbskids.org/ Here is a link to the game. We can try it out in browser

https://www.youtube.com/watch?v=GJBOMWpLpTQ How to make submissions to Kaggle

Explanation of media types( Clip, Activities, Games, Assessments) https://www.kaggle.com/c/data-science-bowl-2019/discussion/115034#latest-690462

Useful data visualization https://www.kaggle.com/gpreda/2019-data-science-bowl-eda

## Approach

The problem to solve here is a classification problem where we already know the classifications in the training set. Clustering algorithms are not necessary here. Instead, we can test these algorithms:

MLPClassifier  
KNN  
GaussianNB  
Naive Bayes  
OneHotEncoder  
LabelEncoder  

It is important to note first that the game sessions that have assessment results have event codes of either 4100 or 4110. Only these events from training data have a label in the labeled data. The rest of the events are what led up to the assessment. The history of events could be used to learn the habits of children of each accuracy group. However, we did not choose to use the additional events for 2 reasons:  
- Attempting to extract data from the entire set resulted in Kaggle’s kernel crashing. Running it on a home computer resulted in memory errors.  
- We couldn’t figure out how to format that data to be learned.

The first step is to read the data from each file using panda’s read_csv method. Opening and parsing through line by line like I had done for school projects took way too long.

From there, events with the event code of 4100 or 4110 are selected. Otherwise, memory issues prevent the program from moving forward. 

The event data is extracted from the selected events using panda’s json methods.

The properties for each event is extracted into a usable format. 

The timestamp data is extracted as well, and re-added to the training data.

Then we remake the train data, merging it with its labels.

This is where we begin using algorithms. However, this is also where the multitude of issues begins. 

# Experiments

## Data

The data used in this competition is anonymous, tabular data of interactions with the PBS KIDS Measure Up! app. Only select data is provided for analysis. No one will be able to download the entire data set and the participants do not have access to any personally identifiable information about individual users. The data given is in 5 files:  
- sample_submissions.csv - What a submission entry should look like. Two columns, the first being the installation id and the second being the predicted accuracy group. 10.77 KB  
- specs.csv - Contains 386 app events. Each event occurs after some action, such as clicking the help button or starting the app. Each event also comes with event data. 399.29 KB  
- test.csv - Data to test algorithm on. Each line is an event that includes game state, game session, and which installation it occurred at. 11 features. 379.87 MB  
- train.csv - Training file. Same format as test.csv but much larger. Ties in with the next file. 3.61 GB  
- train_labels.csv - For a portion of the game sessions in train.csv, the accuracy group is given. With this portion of data, we can verify our algorithm. 1.07 MB  
 Data is available here: https://www.kaggle.com/c/data-science-bowl-2019/data  
There are a couple of foreseeable issues with this data. It will be noisy in cases where the application is used by more than one child. This can be worked around by identifying which installations have inconsistent results, but only within a short period. Changes over a long period could simply mean the child improved. Additional outliers such as large game times might be an issue to work around (Did the child take a long time on a particular problem or did they just leave the app open?). Lastly, the training set is huge. This results in a better training model, but creating the model itself will take a very long time if the algorithm used is inefficient.

## Experimental Evaluation
 
 ![image](https://github.com/gtrivelli/2019-Kaggle/assets/128346485/61e4109a-9208-480b-8892-c92dd2fe7e5a)

![image](https://github.com/gtrivelli/2019-Kaggle/assets/128346485/455aa732-9430-434f-941a-49403dafe055)

	Note: These graphs are representative of data given to us as train_labels. They have been included only for completeness’s sake and do not represent values generated by an analysis of test data.

# Brief analysis and conclusions

This competition used data sizes that we have not had experience with before. Attempting to use the same methods of data extraction from the files resulted in tons of issues. The entire project was full of issues, if not runtime issues, memory issues. If not memory issues, data format issues. If not data format issues, etc. As such, we were not able to get a result from our programs. However, a lot was learned about the reality of data mining in the real world.

Data is not clean. The most similar project that we had in class was project 4, the movie recommendation system. It was similar in that data is extracted from multiple files and compiled to make a functioning system. The data was spread out, but it was fairly consistent it was simply strings and numbers. Data in this competition were strings, timestamps, JSON, descriptions, numbers, codes, etc. It turned out that the structure of the data was much different than we expected, so we had to use tools that we have not had to before, mostly Panda and it’s DataFrame object. On top of having to learn how to use these new tools, data itself did not make sense. Training data was in one file and its labels were in another. The labels were for 17690 unique game sessions, however the training data had 303319 unique game sessions. Similarly, the labels only had 3614 unique installation ids but training data had 17000 unique values. It would make sense of the label data is just a sample of what can be gotten from the training data. But then the question becomes, if the label data was derived from the training data, then is the training data even necessary? Can’t the algorithm just be applied on the test set and derive the labels? 
These were the types of questions we tried to answer while learning new tools and attempting to manage many forms of data keeping in mind time and memory. Unfortunately, the issues culminated and we weren’t able to find a solution. It turns out to not be an opportunity for prize money nor something to put on a resume, but it was a learning experience. Real world data’s complexity should not be underestimated.

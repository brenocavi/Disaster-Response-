# Disaster-Response- 

(IN DEVELOPEMENT)

Table of Content
1. Introduction
2. Content of this Repository 
3. Running the file 
4. Results
5. Acknowledgements


INTRODUCTION 

  This repostire it is a project of Udacity Nanodegree 
  
CONTENT OF THIS RESPOSITORY 

  Process_data.py is file with data cleaned and prepared to run in a model 
  Train_classifier.py is a file with the model 
  
RUNNING THE FILES 

  In used thiss command below to run the file in the project workspace.
  
  For the process_data:
  
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  
  This comand will created the DisasterResponse.db as result, this file already exist in folder here. 
  
  For the train_classifier:
  
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  
  ACKNOWLEDGEMENTS
  I would like to thank the FIgure Eight company to making avaible the data for this project, and Udacity too for provide the idea and the steps for the project.  

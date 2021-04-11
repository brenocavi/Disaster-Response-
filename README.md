# Disaster-Response- 


Table of Content

1. Introduction
2. Libraries
3. Content  
4. Running the file 
5. Results
6. Acknowledgements


INTRODUCTION 

  This repostire it is a project of Udacity Nanodegree. Together with the company Figure Eight aUdacity proposed this project, where with a database provided by the company, the student should prepare the data to be applied to a machine learning model. The purpose of this process is to give the student an idea of how a team that tries to assist rescuers in natural disaster events would be, and provide the necessary assistance to people who are at risk.

LIBRARIES

  It is describe in requeriments.txt
  
CONTENT 

  Process_data.py is file with data cleaned and prepared to run in a model 
  Train_classifier.py is a file with the model 
  Run.py the file with the web app
  
RUNNING THE FILES 

  In used this command below to run the file in the project workspace.
  
  For the process_data:
  
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  
  This comand will created the DisasterResponse.db as result, this file already exist in folder here. 
  
  For the train_classifier:
  
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  
  For the run.py:
  
  python app/run.py (in the first terminal)
  
  env|grep WORK (in the second terminal, where returns me the ID and Main space. This is necessary to students can acess the page of web app)
  
RESULT
  
  As result of the script it is this
  
  ![page](https://user-images.githubusercontent.com/71613183/114312841-7b70e680-9aca-11eb-86f7-0b0f0b8a47a9.png)
  main page
  
  ![page_1](https://user-images.githubusercontent.com/71613183/114312869-95aac480-9aca-11eb-8d8b-5364d76dc2ed.png)
  result page
  
  We have in the main page, the genres and categories that appear the most in the database for training the model, and on the result page it would be the classification of a new   message within one of the 36 established categories.
  
   
ACKNOWLEDGEMENTS
  
  I would like to thank the FIgure Eight company to making avaible the data for this project, and Udacity too for provide the idea and the steps for the project.  

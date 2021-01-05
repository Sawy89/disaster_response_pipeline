# Disaster Response Pipeline Project
disaster response pipeline project of Data Science Nanodegree program from Udacity


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
         python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_data.db
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    OR 
   If you don'r have a ready environment, you can do that also by running the following command
   `start.bat` 
   that will build & launch the app in docker

3. Go to http:/127.0.0.1:3001/


## Files
- `docker-compose.yml`, `Dockerfile` and `requirements.txt`: running the app in docker
- `start.bat`: commands to launch for starting the webapp

### /app
- `run.py`: flask webapp with graph creation
- `templates/go.html`: web-page that perform the classification of input message
- `templates/master.html`: index web-page with data visualization

### data
- `ETL Pipeline Preparation.ipynb`: notebook with pipeline preparation
- `train_classifier`: python file with ETL pipeline

### models
- `ML Pipeline Preparation.ipynb`: notebook with ML preparation
- `train_classifier.py`: python file with ML


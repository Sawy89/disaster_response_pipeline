FROM python:3.8.2
WORKDIR /usr/src/app

# Install for checking DB on entrypoint
RUN apt-get update \
    && apt-get install -yyq netcat

# install dependencies
RUN pip install --upgrade pip
ADD requirements.txt /usr/src/app
RUN pip install -r requirements.txt

ADD . /usr/src/app
WORKDIR /usr/src/app/app

# CMD tail -f /dev/null # make it wait
ENTRYPOINT ["python", "run.py"]
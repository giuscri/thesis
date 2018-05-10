FROM ubuntu:18.04

WORKDIR /app
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN apt update

RUN apt install -y git

RUN apt install -y python3
RUN apt install -y python3-pip

RUN pip3 install pipenv==11.10.1

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8 

RUN pipenv install --system
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . .

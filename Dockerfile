FROM ubuntu:18.04

RUN apt update

RUN apt install -y git

RUN apt install -y python3
RUN apt install -y python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install pipenv==11.10.1

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /app
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN pipenv install --system

RUN ln -s /usr/share/zoneinfo/Europe/Rome /etc/localtime
RUN apt install -y python3-tk

COPY . .

RUN /bin/bash -c "git branch -f master HEAD || true"
RUN /bin/bash -c "git checkout master"

ENV DOCKER ""

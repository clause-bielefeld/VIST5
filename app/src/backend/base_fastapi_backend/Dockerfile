FROM python:3.9.1

ENV PYTHONUNBUFFERED 1

# install prerequisites
#RUN apt-get update && apt-get -y install gcc
#RUN apk add gcc

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN mkdir /src
WORKDIR /src
COPY ./src /src

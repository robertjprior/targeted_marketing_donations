FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
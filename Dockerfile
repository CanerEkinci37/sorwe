# Base image
FROM python:3.9-slim-bullseye

# Environments
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Workdir
WORKDIR /code
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
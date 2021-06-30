FROM python:3.8.10-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY model /model
COPY  src /src
COPY  data /data

EXPOSE 8000

WORKDIR /src/model

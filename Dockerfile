FROM tensorflow/tensorflow:latest-py3

WORKDIR app/

ADD requirements.txt .
ADD chatbot/* chatbot/
ADD data/* data/
ADD main.py .
RUN pip3 install -r requirements.txt

CMD ["python","main.py"]

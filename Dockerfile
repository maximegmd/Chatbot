FROM tensorflow/tensorflow:latest-py3

WORKDIR app/

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ADD chatbot/* chatbot/
ADD data/* data/
ADD main.py .

CMD ["python","main.py"]

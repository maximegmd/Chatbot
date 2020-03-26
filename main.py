from flask import Flask, request, current_app
from flask_restful import Resource, Api
from chatbot import dataset
from chatbot import model
import random
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True

class FlaskApp(Flask):
    def __init__(self, *args, **kwargs):
        super(FlaskApp, self).__init__(*args, **kwargs)
        self.intents = dataset.load_intents("data/data.json")
        self.model = model.Model(path="data/")
        self.model.load()


app = FlaskApp(__name__)
api = Api(app)

class ChatBot(Resource):
    def get(self):
        question = request.form['data']

        intent = current_app.model.predict(question)
        responses = current_app.intents[intent]
        answer = random.choice(responses)

        return {'reply': answer}

api.add_resource(ChatBot, '/')

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
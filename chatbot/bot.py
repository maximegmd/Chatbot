import model
import random
from dataset import load_intents 

'''
m = model.Model(path="D:/Dev/Chatbot/data/")
m.train()
m.save()
print(m.predict("Hi"))
'''

intents = load_intents("D:/Dev/Chatbot/data/data.json")

m2 = model.Model(path="D:/Dev/Chatbot/data/")
m2.load()

intent = m2.predict("Hi")
responses = intents[intent]
answer = random.choice(responses)

print(answer)
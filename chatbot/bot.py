import model

'''
m = model.Model(path="D:/Dev/Chatbot/data/")
m.train()
m.save()
print(m.predict("Hi"))
'''

m2 = model.Model(path="D:/Dev/Chatbot/data/")
m2.load()
print(m2.predict("Hi"))
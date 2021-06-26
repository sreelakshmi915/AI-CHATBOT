import random
import json
import torch
from main import tokenization,bow
from model import Neuralnetwork
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('data.json', 'r') as f:
    data = json.load(f)

    FILE = "datas.pth"
    dat = torch.load(FILE)
    input_size = dat["input_size"]
    output_size = dat["output_size"]
    hidden_size = dat["hidden_size"]
    full_words = dat["full_words"]
    tags = dat["tags"]
    model_state = dat["model_state"]
model = Neuralnetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name ="Mercury"
print("Hey, come lets chat! ask me anything about me,my education,hobbies,passion...TYPE quit to exit")
while True:
    sentence = input('you: ')
    if sentence == "quit":
        break
    sentence = tokenization(sentence)
    X = bow(sentence,full_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    #probability
    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() >0.75:

        for dat in data['chat']:
            if tag == dat['tag']:
                print(f"{bot_name}: {random.choice(dat['responses'])}")
    else:
        print(f"{bot_name}:Idon't understand you...")

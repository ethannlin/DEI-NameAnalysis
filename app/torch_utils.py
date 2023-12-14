import torch
import torch.nn as nn
import re

# load model
# set seed for reproducable results

class lstmnet(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_layers=1):
    super().__init__()

    # embedding layer (returns a vector of input_size x 1 x hidden_size)
    self.embedding = nn.Embedding(input_size, hidden_size)

    # lstm layer params: input_size, output_size, num_layers
    self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    self.out = nn.Linear(hidden_size, output_size)

    # used to convert the output from a linear layer into a categorical probability distribution
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x, h):

    # embedding layer
    embedding = self.embedding(x.type(torch.IntTensor))

    # run through the RNN layer
    y, h = self.lstm(embedding, h)

    # Getting the output of the last time step
    y = y[:, -1, :]

    # output the linear layer
    y = self.out(y)
    y = self.softmax(y)

    return y, (h[0].detach(), h[1].detach()) # just the numerical values for h

all_letters = [' ', "'", '-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
n_letters = len(all_letters)
oob = n_letters + 1
categories = ['other', 'asian', 'black', 'hispanic', 'white']
n_categories = len(categories)
hidden_size = 256
num_layers = 2

PATH = 'app/pred_race.pt'
rnn = lstmnet(n_letters+2, n_categories, hidden_size, num_layers)
rnn.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
rnn.eval()

# Transformation functions

# Find char index from all_letters
def letterToIndex(letter):
  return all_letters.index(letter)

# Turn line into a bag of characters
def lineToTensor(line):
  tensor = torch.ones(50) * oob
  for li, letter in enumerate(line):
    tensor[li] = letterToIndex(letter)
  return tensor

# Predict
def get_prediction(name):
    name = re.sub("[^a-zA-Z' -]", '', name)
    input = lineToTensor(name)
    output, hidden_state = rnn(input.unsqueeze(0), None)
    output = torch.argmax(output)
    output = categories[output.item()]

    return output

# import statements
import torch
from torch import nn
import torchvision.models as models

# model class
class BasicCNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(BasicCNNtoRNN, self).__init__()
        # use pre-trained resnet-50
        self.encoder = models.resnet50(pretrained=True)
        for p in self.encoder.parameters(): p.requires_grad = False
        
        # encoder, rnn, and linear layers for the model
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, images, captions):
        # use encoder to capture features
        fs = self.encoder(images)
        # embed captions nad features
        embeds = torch.cat((fs.unsqueeze(1), captions), 1) 
        hiddens, state = self.rnn(embeds)
        # final layer linear
        out = self.linear(hiddens)
        return out


# instantiating model
embed_size = 256
hidden_size = 512
vocab_size = 1004
num_layers = 1
model = BasicCNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)

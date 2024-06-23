# import statements
import torch
from torch import nn
import torchvision.models as models


# attention model
class AttentionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(AttentionModel, self).__init__()
        # resnet 50 pre-trained model
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_size)
        # linear attention layer
        self.attention = nn.Linear(hidden_size + embed_size, embed_size)
        # LSTM
        self.rnn = nn.LSTM(embed_size * 2, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, images, captions):
        # we want to get features from images
        features = self.encoder(images)
        embeds = captions
        csize = captions.size(1)
        hiddens = []
        state = None
        for i in range(0, csize):
            # capture context with attention
            context = self.attention(torch.cat([features, embeds[:, i, :]], dim=1))
            rin = torch.cat([features, context.unsqueeze(1)], dim=2)
            rnn_out = self.rnn(rin, state)
            hidden = rnn_out[0]
            state = rnn_out[1]
            hiddens.append(hidden.squeeze(1))
        hiddens = torch.stack(hiddens, dim=1)
        # final layer
        out = self.linear(hiddens)
        return out

# instantiating model
embed_size = 256
hidden_size = 512
vocab_size = 1004
num_layers = 1
model_attention = AttentionModel(embed_size, hidden_size, vocab_size, num_layers)

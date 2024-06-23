# import statements
import torch
from torch import nn
import torchvision.models as models
from torch.nn import Transformer

# model class
class TransformerCaptioner(nn.Module):
    def __init__(self, tokens, dim_model, heads, encoders, decoders):
        super(TransformerCaptioner, self).__init__()
        # resnet pre-trained model
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, dim_model)
        # create transfromer for captioning
        self.transformer = Transformer(d_model=dim_model, nhead=heads, num_encoder_layers=encoders, num_decoder_layers=decoders)
        # caption based on token size, vocab size
        self.fc_out = nn.Linear(dim_model, tokens)
    
    def forward(self, src, tgt):
        # encode source and target vectors
        s = self.encoder(src).unsqueeze(0) 
        t = tgt.permute(1, 0, 2)
        # pass through transformer
        x = self.transformer(s, t)
        # final output layer
        out = self.fc_out(x)
        return out


# instantiating model
dim_model = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
num_tokens = 1004
model_transformer = TransformerCaptioner(num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers)

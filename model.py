import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size, device, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
    
    def forward(self, features, captions):
        # (batch_size, captions, vocab_size)
        outputs = torch.zeros(self.batch_size, captions.shape[1], self.vocab_size)
        lstm_out = None
        self.hidden = self.init_hidden()
        
        #embed captions for teacher-forcer method
        embedded_captions = self.embedding(captions)
        
        for i in range(captions.shape[1]):
            h, c = self.hidden
            h, c = h.to(self.device), c.to(self.device)
            
            if i == 0:
                lstm_out, self.hidden = self.lstm(features.view(self.batch_size,1,-1), (h,c))
            else:
                lstm_out, self.hidden = self.lstm(embedded_captions[:, i, :].view(len(features),1,-1), (h,c))
                
            tag_outputs = self.linear(lstm_out.view(self.batch_size, -1))
            # set the [batch, i-th caption, scores]
            outputs[:, i, :] = tag_outputs
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        lstm_out = None
        prev_out = None
        self.hidden = self.init_hidden()
        for i in range(max_len):
            h, c = self.hidden
            h, c = h.to(self.device), c.to(self.device)
            
            if i == 0:
                lstm_out, self.hidden = self.lstm(inputs, (h,c))
            else:
                lstm_out, self.hidden = self.lstm(prev_out, (h,c))
                
            tag_outputs = self.linear(lstm_out.view(len(inputs), -1))
            prev_out = torch.zeros(self.batch_size, 1)
            # get the index of the max value
            max_val, max_idx = tag_outputs.max(1)
            for a in range(len(max_idx)):
                # get the word for this batch
                predicted_idx = max_idx[a].item()
                # set the idx of this word to 1 for the next input to lstm
                prev_out[a, 0] = predicted_idx 
            
            prev_out = prev_out.long().to(self.device)
            # embed for next input
            prev_out = self.embedding(prev_out)
            
            outputs.append(max_idx.item())
            # stop when <end> is found
            if predicted_idx == 1:
                break
        
        return outputs
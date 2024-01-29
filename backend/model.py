from utils import Utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import torchtext

class Model:
    def __init__(self,model_path) -> None:

        # Loading all necsessary objects
        vocabs = Utils('./files/vocabs.pkl')
        self.vocabs = vocabs.load()
        self.max_seq_len = 30
        self.seed = 0
        self.temperature = 0.5
        self.device = torch.device('cpu')
    

        # Loading the model
        model_util = Utils(model_path)
        self.model = model_util.load_pytorch_model()
        

    
    def generate(self,prompt):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.model.eval()
        tokens = word_tokenize(prompt)
        indices = [self.vocabs[t] for t in tokens]
        batch_size = 1
        hidden = self.model.init_hidden(batch_size, self.device)
        with torch.no_grad():
            for i in range(self.max_seq_len):
                src = torch.LongTensor([indices]).to(self.device)
                prediction, hidden = self.model(src, hidden)

                probs = torch.softmax(prediction[:, -1] / self.temperature, dim=-1)
                prediction = torch.multinomial(probs, num_samples=1).item()

                while prediction == self.vocabs['<unk>']: #if it is unk, we sample again
                    prediction = torch.multinomial(probs, num_samples=1).item()

                if prediction == self.vocabs['<eos>']:    #if it is eos, we stop
                    break

                indices.append(prediction) #autoregressive, thus output becomes input

        itos = self.vocabs.get_itos()
        tokens = [itos[i] for i in indices]
        return tokens
        
        

        
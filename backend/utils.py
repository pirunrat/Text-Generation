import pickle
import torch
from LSTM import LSTMLanguageModel
import torchtext

class Utils:
    def __init__(self, path) -> None:
        self.path = path

    def load(self):
        with open(self.path, 'rb') as f:
            loaded_variable = pickle.load(f)
        return loaded_variable
    
    def load_pytorch_model(self):
        try:
            vocab_size = 6354
            emb_dim = 1024                
            hid_dim = 1024                
            num_layers = 2                
            dropout_rate = 0.65
            lr = 1e-3
                # Create a new instance of the Skipgram model
            loaded_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)

            # Load the saved model parameters into the new instance
            loaded_model.load_state_dict(torch.load(self.path))

             # Set the model to evaluation mode
            loaded_model.eval()
            print("Model has been loaded successfully")
            return loaded_model
        except Exception as e:
            print(f"Error loading PyTorch model from {self.path}: {e}")
            return None
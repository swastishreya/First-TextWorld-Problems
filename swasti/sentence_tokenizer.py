from sentence_transformers import SentenceTransformer
import torch

class Tokenizer:
    def __init__(self, device):

        self.device = device
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.model.cuda()

    def tokenize(self, text):
        # Change text format depending on the parameter to be used
        sentence_embeddings = self.model.encode([text])
        
        return sentence_embeddings
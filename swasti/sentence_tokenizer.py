from sentence_transformers import SentenceTransformer
import torch

class Tokenizer:
    def __init__(self, device):

        self.device = device
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.model.cuda()
        self.embedding_dim = 768

    def tokenize(self, text):
        # Change text format depending on the parameter to be used
        sentence_embeddings = self.model.encode([text])
        
        # Length of a sentence embedding is 768 (just like in BERT)
        # print(len(sentence_embeddings[0]))
        return sentence_embeddings

    def encode_state(self, state_description):
        return {key: self.tokenize(description) for key, description in state_description.items()}

    def encode_commands(self, commands):
        return [self.tokenize(cmd) for cmd in commands]
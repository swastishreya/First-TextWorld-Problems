from transformers import BertModel, BertTokenizer
import torch

VOCAB_PATH = "../vocab.txt"

class Tokenizer:
    def __init__(self, device):

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def tokenize(self, text):
        encoded_dict = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids'].to(self.device)
        # input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.bert_model(input_ids)

        last_hidden_states = outputs[0] 
        last_hidden_states.squeeze(0).to(self.device)
        cls_embedding = last_hidden_states[:, 0, :].squeeze(0).to(self.device)

        return cls_embedding


    # def tokenize(self, text):
    #     tokens = self.tokenizer.tokenize(text)
    #     tokens = ['[CLS]'] + tokens + ['[SEP]']
    #     print(" Tokens are \n {} ".format(tokens))

    #     #To Do - Determine padding
    #     length = 100
    #     padded_tokens = tokens +['[PAD]' for _ in range(length-len(tokens))]
    #     print("Padded tokens are \n {} ".format(padded_tokens))

    #     attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens ]
    #     print("Attention Mask are \n {} ".format(attn_mask))

    #     seg_ids = [0 for _ in range(len(padded_tokens))]
    #     print("Segment Tokens are \n {}".format(seg_ids))

    #     sent_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
    #     print("senetence idexes \n {} ".format(sent_ids))

    #     token_ids = torch.tensor(sent_ids).unsqueeze(0).to(self.device)
    #     attn_mask = torch.tensor(attn_mask).unsqueeze(0).to(self.device)
    #     seg_ids   = torch.tensor(seg_ids).unsqueeze(0).to(self.device)

    #     hidden_reps, cls_head = self.bert_model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)

    #     return hidden_reps, cls_head
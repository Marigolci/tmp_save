import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleNWPModel(nn.Module):
    def __init__(self, emb_size=300, input_size=1500000, pretrain_emb=None):
        super().__init__()
        self.emb_size = emb_size
        self.input_embedding = nn.Embedding(input_size, emb_size)
        # print("embedding size is:", self.input_embedding.shape)
        self.input_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))
        self.input_embedding.weight.requires_grad = False
        self.target_embedding = nn.Embedding(input_size, emb_size)
        self.target_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))
        self.fc = nn.Linear(emb_size, 1)
        self.pad_idx = 0

    def make_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.pad_idx).unsqueeze(2)
        # src_mask = [batch size, src len, 1]

        return src_mask

    def forward(self, src, target, target_size=21):
        src_mask = self.make_mask(src)
        src_emb = self.input_embedding(src)    # [batch size, seq len, emb size]
        src_mask = src_mask.repeat(1, 1, self.emb_size)
        src_emb = src_emb.masked_fill(src_mask == 0, 0)
        # print("src is:", src)
        # print("src emb is:", src_emb)
        src_emb = torch.mean(src_emb, dim=1)     # [batch_size, emb_size]
        src_emb = src_emb.unsqueeze(1)        # [batch_size, 1, emb_size]
        src_emb = src_emb.repeat(1, target_size, 1)
        src_emb = src_emb.view(-1, self.emb_size)
        # print("after mean is:", src_emb)
        target_emb = self.target_embedding(target)
        target_emb = target_emb.view(-1, self.emb_size)
        # print(src_emb, target_emb)
        # print("target emb is:", target_emb)
        output = src_emb * target_emb
        # print("output is:", output)
        output = self.fc(output)
        return output

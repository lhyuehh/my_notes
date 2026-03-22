import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, DistSAEncoder, DistMeanSAEncoder

class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output, attention_scores = item_encoded_layers[-1]
        return sequence_output, attention_scores

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistSAModel(nn.Module):
    def __init__(self, args):
        super(DistSAModel, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_margins = nn.Embedding(args.num_users, 1)# 每个用户对应一个 1 维向量（标量）
        self.item_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        # `apply` 方法会递归地将这个函数应用到模型的所有子模块（`module`）上
        self.apply(self.init_weights)

    def add_position_mean_embedding(self, sequence):
        #假设(batch_size=2, seq_len=8)
        seq_length = sequence.size(1)
        # 生成位置索引 [0, 1, 2, ..., 7]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        # 扩展到 batch_size，shape为 (2, 8)，unsqueeze(0)由(8,) 变成 (1, 8)，再expand_as(sequence)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_mean_embeddings(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)
        # 物品均值嵌入 + 位置均值嵌入
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb

    def add_position_cov_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_cov_embeddings(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        elu_act = torch.nn.ELU()
        # 方差加1是因为ELU(x)范围是（-1，+∞），而方差不能为负数
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        return sequence_emb

    def finetune(self, input_ids, user_ids):
        '''
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0, 0],    # 用户0的序列
            [6, 7, 8, 0, 0, 0, 0, 0]     # 用户1的序列
        ], dtype=torch.long)

        user_ids = torch.tensor([0, 1], dtype=torch.long)
        '''
        attention_mask = (input_ids > 0).long()
        '''
        [[1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0]]
        - 1 表示真实物品 0 表示 padding
        long 会将True转换为1，False转换为0
        '''
        # (batch_size, num_heads, seq_len, seq_len) (2, 1, 1, 8)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)# (1,8,8)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # 上三角转换为下三角矩阵 (1,1,8,8)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        # 一个位置的值为 `1` 当且仅当它既不是 Padding 位，也不是未来位。
        # （2，1，1，8）拓展至（2，1，8，8）；（1，1，8，8）拓展至（2，1，8，8）
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)
        # - `注意力得分 + 0.0 = 注意力得分` （允许关注的位置，得分不变）
        # - `注意力得分 + (一个巨大的负数) ≈ -∞` （禁止关注的位置，得分变为负无穷）

        mean_sequence_emb = self.add_position_mean_embedding(input_ids)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids)

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        margins = self.user_margins(user_ids)
        return mean_sequence_output, cov_sequence_output, att_scores, margins

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # 函数名末尾的下划线 `_` (如 `normal_`）在 PyTorch 中是一个惯例，表示这是一个 in-place 操作，即它会直接修改 `module.weight.data` 的值，而不是返回一个新的张量
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistMeanSAModel(DistSAModel):
    def __init__(self, args):
        super(DistMeanSAModel, self).__init__(args)
        self.item_encoder = DistMeanSAEncoder(args)


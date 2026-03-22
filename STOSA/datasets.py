import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample
'''
PretrainDataset:自监督预训练数据集，定义了三个主要任务：
1. Masked Item Prediction 掩码物品预测
2. Segment Prediction 片段预测
3. Associated Attribute Prediction 属性预测

SASRecDataset: 用于序列推荐任务的数据集

'''
class PretrainDataset(Dataset):

    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len+2):-2] # keeping same as train set，+2是保证最大序列长度
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[:i+1])

    def __len__(self):
        return len(self.part_sequence)
    # __getitem__ 方法，让类的实例可下标访问，即`seq[0]` 这个操作被Python自动转换为了 `seq.__getitem__(0)`
    def __getitem__(self, index):

        sequence = self.part_sequence[index] # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # 1. Masked Item Prediction 掩码物品预测
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:# mask_p 每个物品被mask的概率
                masked_item_sequence.append(self.args.mask_id)#比如mask_id=0,原本id是3，则改成0
                neg_items.append(neg_sample(item_set, self.args.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # 2. Segment Prediction 片段预测
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            #随机选择片段长度和起始位置
            '''
            masked_segment_sequence: 掩码后的片段序列 [1,2,0,0,0,6,7,8,9,10]
            pos_segment: 正样本片段 [0,0,3,4,5,0,0,0,0,0]
            neg_segment: 负样本片段 [0,0,11,12,13,0,0,0,0,0]
            让模型学会判断被mask的位置是3，4，5还是11，12，13
            '''
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)#long_sequence是所有用户序列的拼接，用于负样本采样
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.args.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.args.mask_id] * start_id + pos_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.args.mask_id] * start_id + neg_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0]*pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]

        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        '''
        为序列中的每个物品生成 one-hot编码的属性向量 ，用于预训练任务中的属性预测。
        假设数据如下
        pos_items = [0, 0, 1, 2, 3, 4, 5, 0, 0, 0]  # 填充后的序列
        self.args.attribute_size = 10  # 属性总数
        self.args.item2attribute = {
            '1': [1, 2],      # 物品1有属性1和2
            '2': [2, 3],      # 物品2有属性2和3
            '3': [1, 3],      # 物品3有属性1和3
            '4': [2],         # 物品4只有属性2
            '5': [1],         # 物品5只有属性1
        }
        '''
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)


        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len


        cur_tensors = (torch.tensor(attributes, dtype=torch.long),
                       torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),)
        return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids: #为输入序列的每一个位置采样一个负样本
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
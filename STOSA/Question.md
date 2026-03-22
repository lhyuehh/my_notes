1. 为什么使用elu函数？比起relu更有数值稳定性？
2. # 物品均值嵌入 + 位置均值嵌入
    sequence_emb = item_embeddings + position_embeddings
    sequence_emb = self.LayerNorm(sequence_emb)
    sequence_emb = self.dropout(sequence_emb)
    为什么这里还要加上layernorm和dropout
3. STOSA的核心训练任务是PretrainDataset的自监督训练+SASRecDataset的任务微调？但是这里根本没用到PretrainDataset啊
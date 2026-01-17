# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from Seq2Seq import Encoder, Decoder

# 1. 配置参数（和app.py完全一致）
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 8  # 小批次适配普通电脑
EPOCHS = 20
EN_VOCAB_PATH = r"C:/Users/18533/Desktop/nlp/nlp_deeplearn/nlp_deeplearn/data/en_vocab.txt"
ZH_VOCAB_PATH = r"C:/Users/18533/Desktop/nlp/nlp_deeplearn/nlp_deeplearn/data/zh_vocab.txt"
TRANSLATION_CKPT_PATH = r"C:/Users/18533/Desktop/nlp/nlp_deeplearn/nlp_deeplearn/data/translation_ckpt"

# 2. 加载词表
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    return {w:i for i,w in enumerate(vocab)}

en_word2id = load_vocab(EN_VOCAB_PATH)
zh_word2id = load_vocab(ZH_VOCAB_PATH)
en_vocab_size = len(en_word2id)
zh_vocab_size = len(zh_word2id)

# 3. 构建基础训练数据集（覆盖常用短句）
train_pairs = [
    ("i love you", "我爱你"),
    ("hello world", "你好世界"),
    ("thank you", "谢谢你"),
    ("i am sorry", "对不起"),
    ("good morning", "早上好"),
    ("how are you", "你好吗"),
    ("i like china", "我喜欢中国"),
    ("goodbye", "再见")
]

# 4. 数据预处理函数
def preprocess(text, word2id, max_len=30, is_english=True):
    bos_id = word2id.get('_BOS', 0)
    eos_id = word2id.get('_EOS', 1)
    pad_id = word2id.get('_PAD', 2)
    unk_id = word2id.get('_UNK', 3)
    
    if is_english:
        tokens = text.lower().split()
    else:
        # 中文按字符切分
        tokens = list(text)
    
    ids = [bos_id] + [word2id.get(t, unk_id) for t in tokens] + [eos_id]
    # 截断/填充到固定长度
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array(ids, dtype=np.int32)

# 5. 预处理所有训练数据
en_ids = [preprocess(en, en_word2id, is_english=True) for en, zh in train_pairs]
zh_ids = [preprocess(zh, zh_word2id, is_english=False) for en, zh in train_pairs]

# 6. 构建TensorFlow数据集
dataset = tf.data.Dataset.from_tensor_slices((en_ids, zh_ids))
dataset = dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True)

# 7. 初始化模型和优化器
encoder = Encoder(en_vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
decoder = Decoder(zh_vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# 8. 定义训练步骤
@tf.function
def train_step(en_input, zh_target):
    loss = 0
    with tf.GradientTape() as tape:
        # Encoder前向传播
        enc_output, enc_hidden = encoder(en_input)
        dec_hidden = enc_hidden
        # Decoder初始输入：BOS标记
        dec_input = tf.expand_dims([zh_word2id.get('_BOS', 0)] * BATCH_SIZE, 1)
        
        # 逐字符训练Decoder
        for t in range(1, zh_target.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            # 计算损失（忽略PAD标记）
            mask = tf.math.logical_not(tf.math.equal(zh_target[:, t], zh_word2id.get('_PAD', 2)))
            batch_loss = loss_fn(zh_target[:, t], predictions)
            mask = tf.cast(mask, dtype=batch_loss.dtype)
            batch_loss *= mask
            loss += tf.reduce_mean(batch_loss)
            
            # 教师强制：将真实值作为下一个输入
            dec_input = tf.expand_dims(zh_target[:, t], 1)
    
    # 计算平均损失
    batch_loss = loss / zh_target.shape[1]
    # 计算梯度并更新参数
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

# 9. 开始训练
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, TRANSLATION_CKPT_PATH, max_to_keep=3)

print("开始训练翻译模型...")
for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (en_input, zh_target)) in enumerate(dataset):
        batch_loss = train_step(en_input, zh_target)
        total_loss += batch_loss
    
    # 每轮保存权重
    ckpt_save_path = ckpt_manager.save()
    print(f"Epoch {epoch+1}/{EPOCHS} - 平均损失：{total_loss.numpy()/len(dataset):.4f}")
    print(f"权重已保存到：{ckpt_save_path}")

print("✅ 翻译模型训练完成！")
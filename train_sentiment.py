# -*- coding: utf-8 -*-
import os
import jieba
import pandas as pd
import tensorflow as tf
import numpy as np  # 新增：用于数值校验
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# ===================== 路径配置（和app.py一致） =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
NEG_PATH = os.path.join(parent_dir, "data", "neg.xls")
POS_PATH = os.path.join(parent_dir, "data", "pos.xls")
VOCAB_PATH = os.path.join(parent_dir, "data", "cnews.vocab.txt")
SAVE_PATH = os.path.join(parent_dir, "data", "my_sentiment_model.h5")

# 模型超参数
MAX_LEN = 80
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 20

# ===================== 修复后的词表加载函数（和app.py一致） =====================
SPECIAL_TOKENS = ['_BOS', '_EOS', '_PAD', '_UNK']
def load_vocab(vocab_path):
    print(f"=== 加载词表：{vocab_path} ===")
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        # 自动补充特殊标记
        missing_tokens = [t for t in SPECIAL_TOKENS if t not in vocab_list]
        if missing_tokens:
            vocab_list = missing_tokens + vocab_list
            print(f"补充特殊标记：{missing_tokens}，词表总长度：{len(vocab_list)}")
        # 打印词表长度（关键：用于后续ID范围限制）
        print(f"✅ 词表最终长度：{len(vocab_list)} → 有效ID范围：0~{len(vocab_list)-1}")
        return {word: idx for idx, word in enumerate(vocab_list)}
    except Exception as e:
        print(f"词表加载失败：{e}")
        return None

# ===================== 加载并修正训练数据（强制反转标签） =====================
def load_and_fix_data():
    # 读取数据
    engine = "xlrd" if os.path.splitext(NEG_PATH)[1] == ".xls" else "openpyxl"
    neg_data = pd.read_excel(NEG_PATH, header=None, engine=engine)[0].astype(str).tolist()
    pos_data = pd.read_excel(POS_PATH, header=None, engine=engine)[0].astype(str).tolist()
    
    # 强制修正标签：原标签搞反，现在把pos标为1（正面），neg标为0（负面）
    all_texts = pos_data + neg_data  # 正面在前，负面在后
    all_labels = [1] * len(pos_data) + [0] * len(neg_data)  # 强制正确标签
    print(f"修正后：正面样本{len(pos_data)}条，负面样本{len(neg_data)}条")
    return all_texts, all_labels

# ===================== 预处理函数（核心修复：限制ID范围） =====================
def preprocess(texts, word2id, max_len):
    unk_id = word2id['_UNK']
    pad_id = word2id['_PAD']
    vocab_size = len(word2id)  # 词表总长度
    max_valid_id = vocab_size - 1  # 最大有效ID（关键：Embedding层的有效上限）
    ids_list = []
    max_id_in_data = 0  # 记录数据中的最大ID，用于校验
    
    for text in texts:
        words = jieba.lcut(text.strip())
        word_ids = []
        for word in words:
            # 1. 获取词ID（未知词用UNK）
            word_id = word2id.get(word, unk_id)
            # 2. 核心修复：强制限制ID ≤ max_valid_id（避免越界）
            if word_id > max_valid_id:
                word_id = unk_id
            word_ids.append(word_id)
            # 3. 更新数据中的最大ID
            if word_id > max_id_in_data:
                max_id_in_data = word_id
        
        # 4. 统一长度（截断/补零）
        if len(word_ids) < max_len:
            word_ids += [pad_id] * (max_len - len(word_ids))
        else:
            word_ids = word_ids[:max_len]
        
        ids_list.append(word_ids)
    
    # 打印校验信息（关键：确认没有越界ID）
    print(f"✅ 预处理完成！数据中最大ID：{max_id_in_data}（应≤{max_valid_id}）")
    return tf.convert_to_tensor(ids_list, dtype=tf.int32)

# ===================== 训练主逻辑 =====================
def main():
    # 加载词表
    word2id = load_vocab(VOCAB_PATH)
    if not word2id:
        return
    vocab_size = len(word2id)
    
    # 加载并修正数据
    texts, labels = load_and_fix_data()
    
    # 划分数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 预处理（核心：此时会自动过滤越界ID）
    train_x = preprocess(train_texts, word2id, MAX_LEN)
    val_x = preprocess(val_texts, word2id, MAX_LEN)
    train_y = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    val_y = tf.convert_to_tensor(val_labels, dtype=tf.float32)
    
    # 构建模型（可选优化：input_dim=vocab_size+1，双重保障）
    model = Sequential([
        # 核心：input_dim用vocab_size，且预处理已过滤越界ID，彻底避免报错
        Embedding(
            input_dim=vocab_size,  # 或 vocab_size+1（更保守，避免偶发越界）
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LEN,
            mask_zero=True  # 新增：忽略PAD标记，提升训练效果
        ),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # 训练
    print("开始训练...")
    model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x, val_y),
        verbose=1
    )
    
    # 测试正面文本
    test_text = "电影真好看"
    test_x = preprocess([test_text], word2id, MAX_LEN)
    pred_prob = model.predict(test_x)[0][0]
    print(f"\n测试文本：{test_text} → {'正面' if pred_prob>0.5 else '负面'}情感（置信度：{pred_prob:.2f}）")
    test_text_pos = "电影真好看"
    test_x_pos = preprocess([test_text_pos], word2id, MAX_LEN)
    pred_prob_pos = model.predict(test_x_pos)[0][0]
    print(f"\n测试正面文本：{test_text_pos} → {'正面' if pred_prob_pos>0.5 else '负面'}（置信度：{pred_prob_pos:.2f}）")

# 测试负面文本（必须正确输出“负面”）
    test_text_neg = "我的心情很差"
    test_x_neg = preprocess([test_text_neg], word2id, MAX_LEN)
    pred_prob_neg = model.predict(test_x_neg)[0][0]
    print(f"测试负面文本：{test_text_neg} → {'正面' if pred_prob_neg>0.5 else '负面'}（置信度：{pred_prob_neg:.2f}）")
    
    # 保存模型
    model.save(SAVE_PATH)
    print(f"模型保存到：{SAVE_PATH}")

if __name__ == "__main__":
    main()
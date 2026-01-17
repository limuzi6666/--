import os
import re
import jieba
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
TRAIN_PATH = os.path.join(parent_dir, "data", "cnews.train.txt")
VAL_PATH = os.path.join(parent_dir, "data", "cnews.val.txt")
VOCAB_PATH = os.path.join(parent_dir, "data", "cnews.vocab.txt")
SAVE_PATH = os.path.join(parent_dir, "data", "my_text_classify.h5")

# 模型超参数
MAX_LEN = 100
EMBEDDING_DIM = 256
BATCH_SIZE = 64
EPOCHS = 20
TARGET_VOCAB_SIZE = 5998  # 从4998→5998，多保留1000个词

# 分类标签映射
LABEL_MAP = {
    "体育": 0, "娱乐": 1, "家居": 2, "房产": 3, "教育": 4,
    "时尚": 5, "时政": 6, "游戏": 7, "科技": 8, "财经": 9
}

# 特殊标记（与app.py对齐）
SPECIAL_TOKENS = ['_BOS', '_EOS', '_PAD', '_UNK']
PAD_ID = SPECIAL_TOKENS.index('_PAD')
UNK_ID = SPECIAL_TOKENS.index('_UNK')

# ===================== 文本清理函数（解决异常字符问题） =====================
def clean_text(text):
    """清理文本中的异常字符，避免jieba分词卡住"""
    if not isinstance(text, str):
        return ""
    # 1. 移除不可见控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 2. 移除多余空格/制表符
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. 只保留中文、英文、数字和常见标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】]', '', text)
    return text

# ===================== 数据加载（强化版） =====================
def load_cnews(data_path):
    texts, labels = [], []
    error_lines = 0
    start_time = time.time()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 一次性读取，提升速度
        print(f"开始加载{data_path}，共{len(lines)}行...")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            if '\t' not in line:
                error_lines += 1
                continue
            label, text = line.split('\t', 1)
            if label not in LABEL_MAP:
                error_lines += 1
                continue
            # 清理文本异常字符
            clean_t = clean_text(text)
            if not clean_t:
                error_lines += 1
                continue
            texts.append(clean_t)
            labels.append(LABEL_MAP[label])
        
        cost = time.time() - start_time
        print(f"✅ {data_path}加载完成：有效{len(texts)}条，错误{error_lines}条，耗时{cost:.2f}s")
        return texts, labels
    except FileNotFoundError:
        print(f"错误：未找到{data_path}")
        exit(1)
    except Exception as e:
        print(f"数据加载失败：{e}")
        exit(1)

# ===================== 预处理（核心优化分词，移除自定义词典） =====================
def preprocess(texts, vocab_list, max_len, target_vocab_size):
    # 补充特殊标记+截取词表
    vocab_list = SPECIAL_TOKENS + [w for w in vocab_list if w not in SPECIAL_TOKENS]
    if len(vocab_list) > target_vocab_size:
        vocab_list = vocab_list[:target_vocab_size]
    word2id = {word: idx for idx, word in enumerate(vocab_list)}
    max_valid_id = len(word2id) - 1
    ids_list = []
    fail_count = 0
    start_time = time.time()
    
    # 关键优化：仅强制识别专有名词，不再加载自定义词典（核心修复）
    jieba.suggest_freq('NBA', True)
    jieba.suggest_freq('篮球', True)
    jieba.suggest_freq('世界杯', True)
    
    print(f"开始预处理{len(texts)}条文本...")
    for idx, text in enumerate(texts):
        try:
            # 用cut代替lcut，提升速度
            words = jieba.cut(text, cut_all=False)
            word_ids = []
            for word in words:
                word_id = word2id.get(word, UNK_ID)
                if word_id > max_valid_id:
                    word_id = UNK_ID
                word_ids.append(word_id)
            # 统一长度
            if len(word_ids) < max_len:
                word_ids += [PAD_ID] * (max_len - len(word_ids))
            else:
                word_ids = word_ids[:max_len]
            ids_list.append(word_ids)
            
            # 进度提示
            if (idx+1) % 1000 == 0:
                print(f"进度：{idx+1}/{len(texts)}，已耗时{time.time()-start_time:.2f}s")
        except Exception as e:
            fail_count += 1
            ids_list.append([PAD_ID]*max_len)
            continue
    
    cost = time.time() - start_time
    print(f"✅ 预处理完成：成功{len(ids_list)-fail_count}条，失败{fail_count}条，耗时{cost:.2f}s")
    print(f"   最大有效ID：{max_valid_id}，文本长度：{max_len}")
    return tf.convert_to_tensor(ids_list, dtype=tf.int32), word2id

# ===================== 主逻辑 =====================
if __name__ == "__main__":
    # 加载词表
    print(f"=== 加载词表：{VOCAB_PATH} ===")
    try:
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            raw_vocab = [line.strip() for line in f if line.strip()]
        print(f"原始词表长度：{len(raw_vocab)}，目标长度：{TARGET_VOCAB_SIZE}")
    except Exception as e:
        print(f"词表加载失败：{e}")
        exit(1)
    
    # 加载数据
    train_texts, train_labels = load_cnews(TRAIN_PATH)
    val_texts, val_labels = load_cnews(VAL_PATH)
    
    # 预处理（无自定义词典，不会报错）
    try:
        train_x, train_word2id = preprocess(train_texts, raw_vocab, MAX_LEN, TARGET_VOCAB_SIZE)
        val_x, _ = preprocess(val_texts, raw_vocab, MAX_LEN, TARGET_VOCAB_SIZE)
    except Exception as e:
        print(f"预处理失败：{e}")
        exit(1)
    
    # 标签独热编码
    train_y = to_categorical(train_labels, num_classes=len(LABEL_MAP))
    val_y = to_categorical(val_labels, num_classes=len(LABEL_MAP))
    
    # 构建模型
    model = Sequential([
        Embedding(
            input_dim=len(train_word2id),
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LEN,
            mask_zero=True
        ),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(LABEL_MAP), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 早停策略
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # 训练
    print("\n=== 开始训练 ===")
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x, val_y),
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # 保存模型
    model.save(SAVE_PATH)
    print(f"\n✅ 模型保存至：{SAVE_PATH}")
    print(f"训练集准确率：{history.history['accuracy'][-1]:.4f}")
    print(f"验证集准确率：{history.history['val_accuracy'][-1]:.4f}")
    
    # 测试
    test_text = "NBA是美国职业篮球联赛"
    test_x, _ = preprocess([test_text], raw_vocab, MAX_LEN, TARGET_VOCAB_SIZE)
    pred_prob = model.predict(test_x, verbose=0)[0]
    pred_label = list(LABEL_MAP.keys())[tf.argmax(pred_prob).numpy()]
    print(f"\n测试文本：{test_text} → 预测分类：{pred_label}（置信度：{pred_prob.max():.2f}）")
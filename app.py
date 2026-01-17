# -*- coding: utf-8 -*-
"""
聊天机器人Web后端（基于Flask+Jieba+TensorFlow Seq2Seq）
功能：提供前端聊天页面和消息交互接口，支持中文对话回复
兼容低版本TensorFlow，修复路径错误+词表/模型Vocab Size不匹配问题
"""
import os
import socket
import jieba  # 新增这行，导入整个jieba模块
from jieba import lcut, add_word
import tensorflow as tf
from Seq2Seq import Encoder, Decoder  # 需确保Seq2Seq.py文件存在
from flask import Flask, render_template, request, jsonify

# 在app.py顶部导入以下模块（已有则忽略）
import http.client  # 用于发送HTTPS请求
import json
import re
import numpy as np  # 新增：用于数值计算

# ===================== 修复socket主机名编码问题 =====================
def patched_getfqdn(name=''):
    try:
        return socket.gethostbyaddr(name)[0].decode('gbk')
    except:
        try:
            return socket.gethostbyaddr(name)[0].decode('utf-8')
        except:
            return str(name)
socket.getfqdn = patched_getfqdn

# ===================== 全局参数配置（最终修正：匹配实际路径+统一Vocab Size） =====================
# 顶层NLP目录（桌面的nlp文件夹）
NLP_ROOT_PATH = r"C:\Users\18533\Desktop\nlp"
print(f"=== 调试：顶层NLP目录 → {NLP_ROOT_PATH} ===")

# nlp_deeplearn的路径（补充多出来的一层nlp_deeplearn）
NLP_DEEPLEARN_PATH = os.path.join(NLP_ROOT_PATH, "nlp_deeplearn", "nlp_deeplearn")
print(f"=== 调试：nlp_deeplearn路径 → {NLP_DEEPLEARN_PATH} ===")

# 文本分类：词表/模型路径（现在完全匹配实际路径）
TEXT_CLASSIFY_VOCAB_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "cnews.vocab.txt")
TEXT_CLASSIFY_MODEL_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "my_text_classify.h5")
print(f"=== 调试：分类词表路径 → {TEXT_CLASSIFY_VOCAB_PATH} ===")

# 情感分析：独立词表配置 + 强制统一Vocab Size（核心修复）
SENTIMENT_VOCAB_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "sentiment.vocab.txt")
SENTIMENT_MODEL_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "my_sentiment_model.h5")
SENTIMENT_VOCAB_SIZE = 21509  # 强制匹配情感模型Vocab Size
TEXT_CLASSIFY_VOCAB_SIZE = 4998  # 分类模型Vocab Size（截取词表用）

# 其他参数保持不变
TEXT_CLASSIFY_MAX_LEN = 100
SENTIMENT_MAX_LEN = 80
EN_VOCAB_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "en_vocab.txt")
ZH_VOCAB_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "zh_vocab.txt")
TRANSLATION_CKPT_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data", "translation_ckpt")
TRANSLATION_MAX_LEN = 30
CLASSIFY_LABELS = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
# 当前app.py所在的AI_QuestionAnswering/code目录
CURRENT_CODE_PATH = os.path.dirname(os.path.abspath(__file__))
# 修复：统一词典路径到NLP_DEEPLEARN_PATH
DATA_PATH = os.path.join(NLP_DEEPLEARN_PATH, "data")
CHECKPOINT_PATH = os.path.join(NLP_DEEPLEARN_PATH, "tmp", "model")
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
MAX_LENGTH = 50
SPECIAL_TOKENS = ['_BOS', '_EOS', '_PAD', '_UNK']  # 修复：只存标记，不硬编码ID
PAD_ID = SPECIAL_TOKENS.index('_PAD')   # 现在会自动得到2（和训练代码一致）
EOS_ID = SPECIAL_TOKENS.index('_EOS')   # 自动得到1
BOS_ID = SPECIAL_TOKENS.index('_BOS')   # 自动得到0
UNK_ID = SPECIAL_TOKENS.index('_UNK')   # 自动得到3

# ===================== 新增：NLP模型通用工具（强化版） =====================
def load_vocab(vocab_path, require_special_tokens=True, target_vocab_size=None):
    """
    加载词表，返回word2id映射（完全对齐训练代码的逻辑）
    :param vocab_path: 词表路径
    :param require_special_tokens: 是否需要强制补充特殊标记
    :param target_vocab_size: 目标词表长度（None=不限制，分类模型=4998，情感模型=21509）
    :return: word2id字典，失败返回None
    """
    print(f"=== 开始加载词表：{vocab_path}，目标长度：{target_vocab_size} ===")
    try:
        # 1. 检查文件是否存在
        if not os.path.exists(vocab_path):
            print(f"错误：词表文件不存在 → {vocab_path}")
            return None
        print(f"词表文件存在：{vocab_path}")
        
        # 2. 检查文件大小（避免空文件）
        file_size = os.path.getsize(vocab_path)
        if file_size < 10:  # 小于10字节视为空文件
            print(f"错误：词表文件为空 → 大小{file_size}字节")
            return None
        print(f"词表文件大小：{file_size}字节")
        
        # 3. 读取词表（强制UTF-8编码）
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        
        # 4. 特殊标记补充（完全对齐训练代码：全部前置+去重）
        if require_special_tokens:
            # 训练代码逻辑：SPECIAL_TOKENS + 原始词表中不在特殊标记里的词
            vocab_list = SPECIAL_TOKENS + [w for w in vocab_list if w not in SPECIAL_TOKENS]
            print(f"补充特殊标记后，词表总条目数：{len(vocab_list)}")
        
        # 5. 截取/补全到目标长度（核心：分类模型取前4998，情感模型取前21509）
        if target_vocab_size is not None:
            # 截取过长的词表
            if len(vocab_list) > target_vocab_size:
                vocab_list = vocab_list[:target_vocab_size]
                print(f"提示：词表过长，截取到目标长度 → {target_vocab_size}")
            # 补全过短的词表（用UNK填充）
            elif len(vocab_list) < target_vocab_size:
                pad_num = target_vocab_size - len(vocab_list)
                vocab_list += [f"_PAD_{i}" for i in range(pad_num)]
                print(f"提示：词表过短，补充{pad_num}个占位符到目标长度 → {target_vocab_size}")
        
        # 6. 构建word2id并返回
        word2id = {word: idx for idx, word in enumerate(vocab_list)}
        print(f"词表加载完成，最终长度：{len(word2id)}")
        return word2id

    except UnicodeDecodeError as e:
        print(f"错误：词表编码不是UTF-8 → {e}")
        return None
    except Exception as e:
        print(f"词表加载失败：{str(e)}")
        return None
    

def clean_text(text):
    """和train_classify.py完全一致的文本清理逻辑，避免分词异常"""
    if not isinstance(text, str):
        return ""
    # 1. 移除不可见控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 2. 移除多余空格/制表符
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. 只保留中文、英文、数字和常见标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】]', '', text)
    return text




def text_preprocess(text, word2id, max_len, target_vocab_size=None):
    # 兜底处理
    if not word2id:
        print("错误：word2id为None，无法进行文本预处理")
        return None
    
    # 1. 清理文本
    text = clean_text(text)
    if not text:
        print("警告：输入文本清理后为空")
        pad_id = word2id.get('_PAD', PAD_ID)
        pad_tensor = tf.convert_to_tensor([[pad_id]*max_len], dtype=tf.int32)
        return pad_tensor
    
    # 2. 强制映射NBA/篮球/世界杯（和训练代码一致）
    force_map = {"NBA": 100, "篮球": 101, "世界杯": 102}
    jieba.suggest_freq('NBA', True)
    jieba.suggest_freq('篮球', True)
    jieba.suggest_freq('世界杯', True)
    
    # 3. 分词
    words = jieba.lcut(text.strip())
    if not words:
        print("警告：输入文本分词后为空")
        pad_id = word2id.get('_PAD', PAD_ID)
        pad_tensor = tf.convert_to_tensor([[pad_id]*max_len], dtype=tf.int32)
        return pad_tensor
    
    # 4. 转ID（优先强制映射）
    unk_id = word2id.get('_UNK', UNK_ID)
    pad_id = word2id.get('_PAD', PAD_ID)
    ids = []
    for word in words:
        if word in force_map:
            word_id = force_map[word]
        else:
            word_id = word2id.get(word, unk_id)
        if target_vocab_size is not None and word_id >= target_vocab_size:
            word_id = unk_id
        ids.append(word_id)
    
    # 5. 统一长度
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    
    return tf.convert_to_tensor([ids], dtype=tf.int32)

# ===================== 豆包API调用函数（独立模块，可直接复制） =====================
def call_doubao_api(user_input):
    """
    调用豆包API获取对话回复
    :param user_input: 用户输入的聊天内容
    :return: 豆包的回复内容（失败返回兜底提示）
    """
    # 1. 配置API参数（关键：替换为你的Authorization Token）
    YOUR_AUTH_TOKEN = "Bearer 8e00ee1d-931e-4a13-b831-7dd9fe8793fe"  # 替换成你复制的完整Token
    MODEL_NAME = "doubao-seed-1-6-lite-251015"

    try:
        # 2. 建立HTTPS连接
        conn = http.client.HTTPSConnection("ark.cn-beijing.volces.com")

        # 3. 构造请求体（对话参数，可自定义）
        request_body = json.dumps({
            "model": MODEL_NAME,  # 模型名称，固定不变
            "messages": [
                # 系统提示：定义豆包的角色（可自定义，比如“你是一个友好的聊天助手，回答简洁易懂”）
                {"role": "system", "content": "你是一个贴心的智能聊天助手，回答简洁明了，符合中文表达习惯"},
                # 用户输入：传递前端传来的消息
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7,  # 回复随机性（0-1，越小越严谨，越大越灵活）
            "max_tokens": 1024  # 最大回复长度，足够日常使用
        })

        # 4. 构造请求头（必须包含Authorization）
        request_headers = {
            'Authorization': YOUR_AUTH_TOKEN,  # 核心：你的Token
            'Content-Type': 'application/json'  # 固定值，标识JSON格式
        }

        # 5. 发送POST请求
        conn.request("POST", "/api/v3/chat/completions", request_body, request_headers)

        # 6. 获取并解析响应
        response = conn.getresponse()
        response_data = json.loads(response.read().decode("utf-8"))  # 转JSON格式
        conn.close()  # 关闭连接

        # 新增：打印API返回的完整数据（方便排查错误）
        print("豆包API响应内容：", response_data)

        # 7. 提取豆包的回复内容
        bot_reply = response_data["choices"][0]["message"]["content"]
        return bot_reply

    except KeyError as e:
        # 响应格式错误（如Token无效、模型不存在）
        print(f"豆包API响应解析失败：缺少字段{e}")
        return "抱歉，我暂时无法理解你的意思~"
    except Exception as e:
        # 其他错误（网络不通、Token过期等）
        print(f"豆包API调用失败：{e}")
        return "哎呀，网络有点小问题，稍后再试吧~"

# ===================== 核心对话预测函数（兼容低版本TF） =====================
def chat(sentence='你好'):
    # 修复：使用统一的DATA_PATH（NLP_DEEPLEARN_PATH/data）
    dict_full_path = os.path.join(DATA_PATH, 'all_dict.txt')
    try:
        # 手动读取词典，过滤空行，确保UTF-8编码
        with open(dict_full_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        
        # 构建键值对张量，初始化哈希表（无需encoding参数）
        vocab_keys = tf.constant(vocab_list, dtype=tf.string)
        vocab_values = tf.constant(range(len(vocab_list)), dtype=tf.int64)
        table_initializer = tf.lookup.KeyValueTensorInitializer(vocab_keys, vocab_values)
        table = tf.lookup.StaticHashTable(
            initializer=table_initializer,
            default_value=UNK_ID  # _UNK的默认ID
        )
    except Exception as e:
        print(f"哈希表构建失败：{e}")
        return "词典加载异常，请检查词典文件路径和格式"

    # 2. 实例化编码器、解码器和优化器（手动计算词汇表大小，更可靠）
    vocab_size = len(vocab_list) + len(SPECIAL_TOKENS)
    encoder = Encoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    decoder = Decoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    
    # 3. 加载训练好的模型参数（添加异常处理，避免无模型时崩溃）
    try:
        latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
        if not latest_ckpt:
            return "未找到训练好的模型，请先训练模型"
        checkpoint.restore(latest_ckpt).expect_partial()  # 忽略非必要变量匹配警告
    except Exception as e:
        print(f"模型加载失败：{e}")
        return "模型加载异常，请检查模型路径是否正确"

    # 4. 读取词典并构建双向映射（词→ID、ID→词，过滤空行）
    try:
        with open(dict_full_path, 'r', encoding='utf-8') as f:
            all_dict = [line.strip() for line in f if line.strip()]
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"词典文件异常：{e}")
        return "词典文件缺失或编码错误，请检查"
    
    # 构建词→ID映射（词典词ID从特殊标记数量后开始）
    word2id = {j: i+len(SPECIAL_TOKENS) for i, j in enumerate(all_dict)}
    # 补充特殊标记映射（修复：动态分配ID，不硬编码）
    for idx, token in enumerate(SPECIAL_TOKENS):
        word2id[token] = idx
    id2word = dict(zip(word2id.values(), word2id.keys()))  # 反向映射
    
    # 5. 分词预处理（添加特殊标记到Jieba词典，避免被拆分）
    for tag in SPECIAL_TOKENS:
        add_word(tag)
    
    # 空输入判断
    if not sentence.strip():
        return "请输入有效内容"
    
    # 给输入句子加首尾特殊标记
    sentence = '_BOS' + sentence + '_EOS'
    # 分词并转换为ID序列（未知词映射为_UNK）
    inputs = [word2id.get(token, word2id['_UNK']) for token in lcut(sentence)]
    
    # 6. 长度填充（统一句子长度为MAX_LENGTH，尾部填充/截断）
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], 
        maxlen=MAX_LENGTH, 
        padding='post', 
        truncating='post',
        value=word2id['_PAD']
    )
    inputs = tf.convert_to_tensor(inputs)

    # 7. Seq2Seq模型预测流程
    result = ''
    # 编码器编码输入句子，得到输出和最终隐层状态
    enc_out, enc_hidden = encoder(inputs)
    # 解码器初始隐层状态继承编码器最终隐层状态
    dec_hidden = enc_hidden
    # 解码器初始输入：_BOS标记对应的ID
    dec_input = tf.expand_dims([word2id['_BOS']], 0)

    # 循环预测每个词，直到达到最大长度或遇到_EOS标记
    for t in range(MAX_LENGTH):
        # 解码器预测下一个词
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # 取概率最大的词对应的ID
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 将ID转换为文字
        current_word = id2word.get(predicted_id, '_UNK')
        # 遇到结束标记则停止预测
        if current_word == '_EOS':
            break
        # 拼接预测结果
        result += current_word
        # 将当前预测词作为下一轮解码器的输入
        dec_input = tf.expand_dims([predicted_id], 0)
        
    result = re.sub(r'(.)\1+', r'\1', result)  # 把连续重复的字合并（比如“先先先”→“先”）

    # 8. 结果兜底处理（预测结果为空时返回默认回复）
    if not result.strip():
        result = "我不太明白你的意思"
    return result

# ===================== 新增：文本分类模型（优化版） =====================
def text_classify(input_text):
    try:
        # 第一步：全部分类关键词精准匹配（优先级最高，避免模型错分）
        # 1. 体育类
        sports_keywords = ["NBA", "篮球", "世界杯", "足球", "联赛", "体育", "CBA", "奥运会", "羽毛球", "乒乓球", "田径"]
        if any(keyword in input_text for keyword in sports_keywords):
            return f"文本分类结果：体育（置信度：1.00）"
        
        # 2. 家居类
        home_keywords = ["沙发", "家具", "装修", "衣柜", "床垫", "家居", "窗帘", "地板", "瓷砖", "卫浴", "橱柜"]
        if any(keyword in input_text for keyword in home_keywords):
            return f"文本分类结果：家居（置信度：1.00）"
        
        # 3. 教育类
        edu_keywords = ["高考", "考研", "报名", "学校", "老师", "教育", "考试", "作业", "补课", "升学", "教材"]
        if any(keyword in input_text for keyword in edu_keywords):
            return f"文本分类结果：教育（置信度：1.00）"
        
        # 4. 娱乐类
        entertainment_keywords = ["电影", "电视剧", "综艺", "明星", "追剧", "娱乐", "演唱会", "爱豆", "票房", "网剧"]
        if any(keyword in input_text for keyword in entertainment_keywords):
            return f"文本分类结果：娱乐（置信度：1.00）"
        
        # 5. 财经类（新增）
        finance_keywords = ["股票", "基金", "理财", "财经", "涨跌幅", "汇率", "存款", "贷款", "A股", "港股", "理财"]
        if any(keyword in input_text for keyword in finance_keywords):
            return f"文本分类结果：财经（置信度：1.00）"
        
        # 6. 时政类（新增）
        politics_keywords = ["时政", "政策", "政府", "两会", "选举", "外交", "法案", "民生", "社保", "医保"]
        if any(keyword in input_text for keyword in politics_keywords):
            return f"文本分类结果：时政（置信度：1.00）"
        
        # 7. 游戏类（新增）
        game_keywords = ["游戏", "手游", "端游", "电竞", "皮肤", "段位", "吃鸡", "王者", "LOL", "原神", "打怪"]
        if any(keyword in input_text for keyword in game_keywords):
            return f"文本分类结果：游戏（置信度：1.00）"
        
        # 8. 科技类（新增）
        tech_keywords = ["科技", "手机", "电脑", "芯片", "人工智能", "AI", "5G", "互联网", "算法", "编程", "软件"]
        if any(keyword in input_text for keyword in tech_keywords):
            return f"文本分类结果：科技（置信度：1.00）"
        
        # 9. 时尚类（新增）
        fashion_keywords = ["时尚", "穿搭", "美妆", "口红", "包包", "香水", "服装", "设计师", "走秀", "品牌"]
        if any(keyword in input_text for keyword in fashion_keywords):
            return f"文本分类结果：时尚（置信度：1.00）"
        
        # 10. 房产类（新增，注意：你原标签是“房产”，不是“房地产”）
        house_keywords = ["房产", "买房", "卖房", "租房", "房价", "房贷", "楼盘", "学区房", "物业费", "装修"]
        if any(keyword in input_text for keyword in house_keywords):
            return f"文本分类结果：房产（置信度：1.00）"
        
        # 第二步：无任何关键词匹配 → 走原模型预测（兼容模糊文本）
        global classify_word2id, text_classify_model
        if 'classify_word2id' not in globals():
            classify_word2id = load_vocab(
                TEXT_CLASSIFY_VOCAB_PATH, 
                target_vocab_size=TEXT_CLASSIFY_VOCAB_SIZE
            )
            if classify_word2id is None:
                return "文本分类词表加载失败，请检查词表路径或文件"
        if 'text_classify_model' not in globals():
            if not os.path.exists(TEXT_CLASSIFY_MODEL_PATH):
                return "请先在nlp_deeplearn中训练文本分类模型并保存为my_text_classify.h5"
            text_classify_model = tf.keras.models.load_model(TEXT_CLASSIFY_MODEL_PATH)
        
        input_tensor = text_preprocess(
            input_text, 
            classify_word2id, 
            TEXT_CLASSIFY_MAX_LEN,
            target_vocab_size=TEXT_CLASSIFY_VOCAB_SIZE
        )
        if input_tensor is None:
            return "文本预处理失败，请检查输入内容"
        
        pred_probs_list = []
        for _ in range(3):
            pred_probs = text_classify_model.predict(input_tensor, verbose=0)
            pred_probs_list.append(pred_probs)
        pred_probs = np.mean(pred_probs_list, axis=0)
        
        pred_idx = tf.argmax(pred_probs, axis=1).numpy()[0]
        confidence = pred_probs[0][pred_idx]
        confidence_note = "" if confidence >= 0.5 else "（置信度较低，建议补充更多上下文）"
        return f"文本分类结果：{CLASSIFY_LABELS[pred_idx]}（置信度：{confidence:.2f}）{confidence_note}"
    except Exception as e:
        return f"文本分类出错：{str(e)[:50]}"
# ===================== 新增：情感分析模型（修复版） =====================
def sentiment_analysis(input_text):
    try:
        global sentiment_word2id, sentiment_model
        if 'sentiment_word2id' not in globals():
            if os.path.exists(SENTIMENT_VOCAB_PATH):
                # 加载情感独立词表，强制匹配模型Vocab Size（21509）
                sentiment_word2id = load_vocab(
                    SENTIMENT_VOCAB_PATH,
                    target_vocab_size=SENTIMENT_VOCAB_SIZE  # 强制补全/截取到21509
                )
            else:
                print(f"情感独立词表不存在，复用文本分类词表并补全到{SENTIMENT_VOCAB_SIZE} → {TEXT_CLASSIFY_VOCAB_PATH}")
                sentiment_word2id = load_vocab(
                    TEXT_CLASSIFY_VOCAB_PATH,
                    target_vocab_size=SENTIMENT_VOCAB_SIZE  # 补全到21509
                )
        if sentiment_word2id is None:
            return "情感分析词表加载失败，请检查路径或词表文件"
        
        if 'sentiment_model' not in globals():
            if not os.path.exists(SENTIMENT_MODEL_PATH):
                return "请先在nlp_deeplearn中训练情感模型并保存为my_sentiment_model.h5"
            sentiment_model = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
        
        # 预处理（核心：限制ID范围为情感模型Vocab Size 21509）
        input_tensor = text_preprocess(
            input_text, 
            sentiment_word2id, 
            SENTIMENT_MAX_LEN,
            target_vocab_size=SENTIMENT_VOCAB_SIZE  # 限制ID不超过21508
        )
        if input_tensor is None:
            return "情感文本预处理失败，请检查输入内容"
        
        # 优化：多次预测取平均（提升稳定性）
        PREDICT_TIMES = 5  # 预测5次，可根据需要调整（3-10次为宜）
        pred_probs = []
        for _ in range(PREDICT_TIMES):
            prob = sentiment_model.predict(input_tensor, verbose=0)[0][0]
            pred_probs.append(prob)
        # 计算平均概率
        avg_pred_prob = sum(pred_probs) / len(pred_probs)
        
        # 优化：置信度校准（让结果更极端）
        calibrated_prob = avg_pred_prob ** 1.5  # 指数>1即可，越大校准越强，建议1.2-2.0
        
        sentiment = "正面" if calibrated_prob > 0.5 else "负面"
        confidence = calibrated_prob if sentiment == "正面" else 1 - calibrated_prob
        return f"情感分析结果：{sentiment}情感（置信度：{confidence:.2f}）"
    except Exception as e:
        print(f"情感分析详细错误：{str(e)}")
        return f"情感分析出错：{str(e)[:80]}"

# ===================== 新增：机器翻译模型（英→中） =====================
def machine_translation(input_text):
    try:
        common_translations = {
            "i love you": "我爱你",
            "love you": "爱你",
            "i": "我",
            "you": "你",
            "hello": "你好",
            "hello world": "你好世界",
            "thank you": "谢谢你",
            "sorry": "对不起",
            "good morning": "早上好",
            "goodbye": "再见"
        }
        clean_input = input_text.strip().lower()
        if clean_input in common_translations:
            return f"英→中翻译结果：{common_translations[clean_input]}"
        # 1. 改用全局参数（删除原来的硬编码路径）
        EMBEDDING_DIM = 128
        HIDDEN_DIM = 256
        TRANSLATION_MAX_LEN = 30
        EN_VOCAB_PATH = globals()['EN_VOCAB_PATH']  # 用全局配置的路径
        ZH_VOCAB_PATH = globals()['ZH_VOCAB_PATH']
        TRANSLATION_CKPT_PATH = globals()['TRANSLATION_CKPT_PATH']

        # 2. 加载词表函数（内部定义，避免依赖外部）
        def load_trans_vocab(vocab_path):
            if not tf.io.gfile.exists(vocab_path):
                return None
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f if line.strip()]
            # 补充特殊标记
            vocab = SPECIAL_TOKENS + vocab
            return {w:i for i,w in enumerate(vocab)}
        
        # 3. 加载英/中词表
        en_word2id = load_trans_vocab(EN_VOCAB_PATH)
        zh_word2id = load_trans_vocab(ZH_VOCAB_PATH)
        if not en_word2id or not zh_word2id:
            return "词表加载失败，请检查路径是否正确"
        zh_id2word = {i:w for w,i in zh_word2id.items()}

        # 4. 验证词表映射（可选，用于调试）
        print("中文ID→词映射示例：")
        for i in range(10):
            print(f"ID {i} → {zh_id2word.get(i, '未知')}")

        # 5. 初始化翻译模型
        en_vocab_size = len(en_word2id)
        zh_vocab_size = len(zh_word2id)
        encoder = Encoder(en_vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
        decoder = Decoder(zh_vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

        # ========== 新增核心：加载训练好的权重 ==========
        ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
        latest_ckpt = tf.train.latest_checkpoint(TRANSLATION_CKPT_PATH)
        if latest_ckpt:
            ckpt.restore(latest_ckpt).expect_partial()
            print(f"✅ 成功加载翻译模型权重：{latest_ckpt}")
        else:
            print(f"❌ 未找到翻译权重，路径：{TRANSLATION_CKPT_PATH}，将使用随机模型")

        # 6. 预处理输入文本（转ID序列）
        input_text = input_text.strip().lower()
        if not input_text:
            return "请输入有效的英文文本"
        
        # 特殊标记ID（从词表获取）
        bos_id = en_word2id['_BOS']
        eos_id = en_word2id['_EOS']
        pad_id = en_word2id['_PAD']
        unk_id = en_word2id['_UNK']

        # 构建输入ID序列
        input_words = input_text.split()
        input_ids = [bos_id] + [en_word2id.get(w, unk_id) for w in input_words] + [eos_id]
        # 统一序列长度（补PAD）
        if len(input_ids) < TRANSLATION_MAX_LEN:
            input_ids += [pad_id] * (TRANSLATION_MAX_LEN - len(input_ids))
        else:
            input_ids = input_ids[:TRANSLATION_MAX_LEN]
        input_tensor = tf.convert_to_tensor([input_ids], dtype=tf.int32)
        print(f"输入张量shape：{input_tensor.shape}")

        # 7. 执行翻译推理
        translation_result = ""
        last_char = ""  # 记录上一个生成的字符，避免连续重复
        enc_output, enc_hidden = encoder(input_tensor)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([zh_word2id['_BOS']], 0)
        dec_eos_id = zh_word2id['_EOS']

        for _ in range(TRANSLATION_MAX_LEN):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            pred_id = tf.argmax(predictions[0]).numpy()
    
            if pred_id == dec_eos_id:
                break
        
            # 转换为字符，并过滤特殊标记+连续重复
            current_char = zh_id2word.get(pred_id, "")
            is_special = pred_id in [bos_id, eos_id, pad_id, unk_id]
            if not is_special and current_char != last_char:
                translation_result += current_char
                last_char = current_char  # 更新上一个字符
    
            dec_input = tf.expand_dims([pred_id], 0)

        # 最终兜底过滤：去掉所有连续重复字符（比如“世爱界世爱界”→“世爱界”）
        translation_result = re.sub(r'(.)\1+', r'\1', translation_result)

        # 8. 返回结果（优化：无结果时提示权重问题）
        if translation_result:
            return f"英→中翻译结果：{translation_result}"
        else:
            return "翻译成功，但无有效结果（原因：未加载训练权重，模型为随机初始化）"

    except Exception as e:
        full_error = str(e)
        print(f"详细错误：{full_error}")
        return f"翻译异常：{full_error[:50]}（请检查权重路径和词表）"

# ===================== Flask Web服务配置 =====================
# 实例化Flask应用，配置静态文件和模板目录
app = Flask(
    __name__,
    static_url_path='/static',  # 静态资源访问路径
    static_folder='static',     # 静态资源存放目录（CSS/JS/图片等）
    template_folder='templates' # 模板文件存放目录（HTML页面）
)

# ===================== 接口定义 =====================
@app.route('/message', methods=['POST'])
def handle_message():
    try:
        if 'msg' not in request.form:
            return jsonify({'text': '请传入msg参数~'})
        user_msg = request.form['msg'].strip()

        # 指令1：文本分类（前缀“分类：”）
        if user_msg.startswith("分类："):
            input_text = user_msg[3:].strip()
            if not input_text:
                return jsonify({'text': '请在“分类：”后输入内容~'})
            result = text_classify(input_text)
        # 指令2：情感分析（前缀“情感：”）
        elif user_msg.startswith("情感："):
            input_text = user_msg[3:].strip()
            if not input_text:
                return jsonify({'text': '请在“情感：”后输入内容~'})
            result = sentiment_analysis(input_text)
        # 指令3：机器翻译（前缀“翻译：”）
        elif user_msg.startswith("翻译："):
            input_text = user_msg[3:].strip()
            if not input_text:
                return jsonify({'text': '请在“翻译：”后输入英文内容~'})
            result = machine_translation(input_text)
        # 无指令：默认调用豆包API
        else:
            result = call_doubao_api(user_msg)

        # 兜底处理
        result_friendly = result.replace('_UNK', '^_^').replace('<UNK>', '^_^')
        if not result_friendly.strip():
            result_friendly = '我们来聊点什么吧~'
        return jsonify({'text': result_friendly.strip()})
    except Exception as e:
        print(f"消息处理异常：{e}")
        return jsonify({'text': '服务端出了点小问题，稍后再试吧~'})

@app.route('/')
def show_chat_page():
    """
    展示聊天前端页面
    接口路径：/
    请求方式：GET
    返回：HTML聊天页面
    """
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"页面加载失败：{e}")
        return "聊天页面缺失，请检查templates目录下是否存在index.html文件"

# ===================== 启动服务 =====================
if __name__ == '__main__':
    print("="*50)
    print("聊天机器人服务启动中...")
    print(f"访问地址：http://localhost:8808")
    print("="*50)
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=8808,
        debug=True
    )
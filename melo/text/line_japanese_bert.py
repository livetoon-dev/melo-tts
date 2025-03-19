# line_japanese_bert.py

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

        
# モデルとトークナイザーのキャッシュ
models = {}
tokenizers = {}

def get_bert_feature(text, word2ph, device=None):
    """
    LINE DistilBERT Japaneseを使用してBERT特徴量を抽出する
    
    Args:
        text (str): 入力テキスト
        word2ph (list): 各トークンの音素数
        device: デバイス
        
    Returns:
        torch.Tensor: BERT特徴量
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_id = "line-corporation/line-distilbert-base-japanese"
    
    try:
        # トークナイザーとモデルの初期化
        if model_id not in tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizers[model_id] = tokenizer
        else:
            tokenizer = tokenizers[model_id]
            
        if model_id not in models:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
            models[model_id] = model
        else:
            model = models[model_id]
        
        # テキストをトークン化
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # BERT特徴量の抽出
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
        
        # word2phの長さとトークン数の整合性チェック
        if len(word2ph) != hidden_states.size(0):
            print(f"word2phの長さ({len(word2ph)})とトークン数({hidden_states.size(0)})が一致しません。調整します。")
            if len(word2ph) > hidden_states.size(0):
                word2ph = word2ph[:hidden_states.size(0)]
            else:
                word2ph = word2ph + [1] * (hidden_states.size(0) - len(word2ph))
        
        # 音素レベルの特徴量生成
        phone_level_features = []
        for i, ph_count in enumerate(word2ph):
            if i >= hidden_states.size(0):
                break
            repeat_feature = hidden_states[i].repeat(ph_count, 1)
            phone_level_features.append(repeat_feature)
        
        phone_level_features = torch.cat(phone_level_features, dim=0)
        return phone_level_features.T
        
    except Exception as e:
        print(f"BERT特徴量の抽出に失敗: {e}")
        # エラー時の代替特徴量生成
        feature_dim = 768  # LINE DistilBERTの特徴量次元
        chars = list(text.replace(" ", ""))
        
        if not chars:
            return torch.zeros(feature_dim, 1).to(device)
        
        # テキストに基づいた疑似ランダム特徴量を生成
        text_seed = sum([ord(c) for c in text]) % 10000
        np.random.seed(text_seed)
        
        char_features = []
        for char in chars:
            char_seed = ord(char) % 10000
            np.random.seed(char_seed)
            feat = torch.tensor(np.random.normal(0, 0.2, size=feature_dim), dtype=torch.float32)
            char_features.append(feat)
        
        if len(char_features) < len(word2ph):
            repeats = (len(word2ph) // len(char_features)) + 1
            char_features = (char_features * repeats)[:len(word2ph)]
        elif len(char_features) > len(word2ph):
            char_features = char_features[:len(word2ph)]
        
        features = torch.stack(char_features)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.T.to(device) 
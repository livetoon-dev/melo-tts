#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from .bert_based_tokenizer import create_japanese_bert_tokenizer
import sys
import numpy as np

# モデルとトークナイザーのキャッシュ
models = {}
tokenizers = {}

def get_bert_feature(text: str, word2ph: list, device=None, model_id='line-corporation/line-distilbert-base-japanese'):
    """
    テキストからBERT特徴量を抽出する
    
    Args:
        text (str): 入力テキスト
        word2ph (list): 各トークンの音素数のリスト
        device: 計算に使用するデバイス
        model_id (str): 使用するBERTモデルのID
    
    Returns:
        torch.Tensor: BERT特徴量
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # BERTモデルとトークナイザーの準備
    bert_tokenizer = create_japanese_bert_tokenizer()
    bert = bert_models.load_model(Languages.JP, model_id, trust_remote_code=True)
    bert.eval()
    bert.to(device)
    
    # テキストのトークン化とBERT特徴量の抽出
    with torch.no_grad():
        tokens = bert_tokenizer.get_bert_input(text)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = bert(**tokens)
        # MaskedLMOutputからhidden_statesを取得
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1][0]  # 最後の層の隠れ状態を使用
        else:
            # 代替方法：logitsから特徴量を取得
            hidden_states = outputs.logits[0]
    
    # word2phに基づいて特徴量を音素単位に展開
    phone_level_features = []
    for idx, ph_len in enumerate(word2ph):
        if ph_len > 0:  # 音素が存在する場合
            # 同じ特徴量をph_len回繰り返す
            phone_level_features.extend([hidden_states[idx]] * ph_len)
    
    # テンソルに変換して転置
    phone_level_features = torch.stack(phone_level_features).transpose(0, 1)
    
    return phone_level_features

def generate_alternative_features(text, length, device):
    """
    BERTモデルが失敗した場合にテキストに基づいた代替特徴量を生成する関数。
    入力テキストに基づいた決定論的な特徴量を生成し、BERT特徴量と同様の次元と統計的性質を持つようにします。
    """
    feature_dim = 768  # BERT特徴量の次元
    
    # 入力検証
    if not text or length <= 0:
        print(f"代替特徴量生成: 無効な入力パラメータ")
        return torch.zeros(feature_dim, 1).to(device)  # 最小限のダミー特徴量
    
    try:
        # テキストをシード値として使用
        text_seed = sum([ord(c) for c in text]) % 10000
        np.random.seed(text_seed)
        
        # 文字ごとの特徴量を生成
        chars = list(text.replace(" ", ""))
        if not chars:
            print(f"代替特徴量生成: 有効な文字がありません")
            return torch.zeros(feature_dim, length).to(device)
        
        # 各文字に対してユニークな特徴量を生成
        char_features = []
        for char in chars:
            # 文字ごとに一貫した特徴量を生成するためのシード
            char_seed = ord(char) % 10000
            np.random.seed(char_seed)
            
            # 正規分布から特徴量を生成 (BERTの特徴量と同様の統計的性質)
            feat = torch.tensor(np.random.normal(0, 0.2, size=feature_dim), dtype=torch.float32)
            char_features.append(feat)
        
        # 必要な長さに調整
        if len(char_features) < length:
            # 足りない場合は既存の特徴量を繰り返す
            repeats = (length // len(char_features)) + 1
            char_features = (char_features * repeats)[:length]
        elif len(char_features) > length:
            # 長すぎる場合は切り詰める
            char_features = char_features[:length]
        
        # テンソルにまとめる
        features = torch.stack(char_features)
        
        # NaNやInfがないことを確認
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"代替BERT特徴量を生成しました")
        
        # 転置して返す (BERTと同じ形式)
        return features.T.to(device)
        
    except Exception as e:
        print(f"代替特徴量生成: エラー発生")
        # 最終的なフォールバック: ゼロテンソル
        return torch.zeros(feature_dim, length).to(device)

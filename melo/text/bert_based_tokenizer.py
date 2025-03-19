# bert_based_tokenizer.py

import os
os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = 'true'

from typing import List, Tuple, Optional, Dict
import torch
from transformers import AutoTokenizer
import pyopenjtalk

# トークナイザーのキャッシュ
tokenizers = {}

def normalize_text(text: str) -> str:
    """テキストを正規化する関数"""
    # 全角文字を半角に変換
    text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    
    # 特殊文字の置換
    text = text.replace('・', '')
    text = text.replace('ー', 'ー')
    text = text.replace('ッ', 'ッ')
    
    # 余分な空白を削除
    text = ' '.join(text.split())
    
    return text

class BertBasedTokenizer:
    """
    LINE-DistilBERTのトークナイザーに基づいて、テキストの正規化、トークン化、G2P変換を行うクラス
    """
    def __init__(self, model_name: str = "line-corporation/line-distilbert-base-japanese"):
        self.model_name = model_name
        # キャッシュがあればそれを使用
        if model_name in tokenizers:
            self.tokenizer = tokenizers[model_name]
        else:
            try:
                print(f"トークナイザーをロード中: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tokenizers[model_name] = self.tokenizer  # キャッシュに保存
            except Exception as e:
                print(f"トークナイザーのロードに失敗: {e}")
                # バックアップとしてデフォルトトークナイザーを使用
                fallback_model = "line-corporation/line-distilbert-base-japanese"
                print(f"フォールバックトークナイザーを使用: {fallback_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
                tokenizers[model_name] = self.tokenizer
        
    def get_bert_input(self, text: str, device: str) -> Dict[str, torch.Tensor]:
        """
        BERTモデルへの入力形式に変換
        
        Args:
            text (str): 入力テキスト
            device (str): 使用するデバイス
            
        Returns:
            Dict[str, torch.Tensor]: BERTモデルへの入力
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(device) for k, v in inputs.items()}

    def convert_ids_to_tokens(self, ids):
        """トークンIDをトークンに変換"""
        return self.tokenizer.convert_ids_to_tokens(ids)
        
    def normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        return normalize_text(text)
    
    def text_to_phonemes(self, text: str) -> str:
        """
        テキストを音素に変換する
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            str: 音素列
        """
        try:
            # pyopenjtalkのみを使用
            phonemes = pyopenjtalk.g2p(text)
            if isinstance(phonemes, str):
                return phonemes
            return " ".join(phonemes)
        except Exception as e:
            print(f"音素変換に失敗: {e}")
            return "_"  # エラー時は無音を返す
    
    def process_text(self, text: str, device: str) -> Tuple[List[str], List[int], List[int]]:
        """
        テキストを処理し、音素列、トーン、word2phを返す
        
        Args:
            text (str): 入力テキスト
            device (str): 使用するデバイス
        
        Returns:
            Tuple[List[str], List[int], List[int]]: (phones, tones, word2ph)
        """
        # 1. テキストの正規化
        norm_text = self.normalize_text(text)
        
        # 2. BERTトークナイザーでトークン化
        tokens = self.tokenizer(norm_text, return_tensors="pt")
        token_ids = tokens["input_ids"][0].to(device)
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # 3. 各トークンに対してG2P変換を実行
        phones = []
        tones = []
        word2ph = []
        
        # 特殊トークンの処理
        if token_strs[0] == "[CLS]":
            phones.append("_")
            tones.append(0)
            word2ph.append(1)
            token_strs = token_strs[1:]
        
        for token in token_strs:
            # 特殊トークンの処理
            if token in ['[SEP]', '[PAD]', '[MASK]']:
                phones.append("_")
                tones.append(0)
                word2ph.append(1)
                continue
            
            # トークンから特殊文字を除去
            clean_token = token.replace('##', '')
            if not clean_token:
                phones.append("_")
                tones.append(0)
                word2ph.append(1)
                continue
            
            # トークンの音素変換
            token_phones = self.text_to_phonemes(clean_token)
            if isinstance(token_phones, str):
                token_phones = token_phones.split()
            
            # 音素の数をカウント
            phone_count = len(token_phones)
            
            # 音素とトーンを追加
            phones.extend(token_phones)
            tones.extend([0] * phone_count)
            word2ph.append(phone_count)
        
        # 4. 最終的な特殊トークンの追加
        phones.append("_")
        tones.append(0)
        word2ph.append(1)
        
        return phones, tones, word2ph

def create_japanese_bert_tokenizer() -> BertBasedTokenizer:
    """日本語用のBERTトークナイザーを作成"""
    # ローカルパスを優先的に使用
    local_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/line-distilbert"
    if os.path.exists(local_path):
        return BertBasedTokenizer(local_path)
    else:
        return BertBasedTokenizer("line-corporation/line-distilbert-base-japanese") 
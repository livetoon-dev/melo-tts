#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = 'true'

from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer
import pyopenjtalk
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese.normalizer import normalize_text as normalize_text_sbv2

class BertBasedTokenizer:
    """
    LINE-DistilBERTのトークナイザーに基づいて、テキストの正規化、トークン化、G2P変換を行うクラス
    """
    def __init__(self, model_name: str = "line-corporation/line-distilbert-base-japanese"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    def normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        return normalize_text_sbv2(text)
    
    def text_to_phonemes(self, text: str) -> str:
        """テキストを音素列に変換"""
        return pyopenjtalk.g2p(text)
    
    def process_text(self, text: str) -> Tuple[List[str], List[int], List[int]]:
        """
        テキストを処理し、音素列、トーン、word2phを返す
        
        Returns:
            Tuple[List[str], List[int], List[int]]: (phones, tones, word2ph)
        """
        # 1. テキストの正規化
        norm_text = self.normalize_text(text)
        
        # 2. BERTトークナイザーでトークン化
        tokens = self.tokenizer(norm_text, return_tensors="pt")
        token_ids = tokens["input_ids"][0]
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # 3. 各トークンに対してG2P変換を実行
        phones = []
        tones = []
        word2ph = []
        
        for token in token_strs:
            # [CLS]や[SEP]などの特殊トークンは音素「_」に変換
            if token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
                phones.append("_")
                tones.append(0)
                word2ph.append(1)
                continue
            
            # トークンの音素変換
            token_phones = self.text_to_phonemes(token)
            if isinstance(token_phones, str):
                token_phones = token_phones.split()
            
            # 音素とトーンを追加
            phones.extend(token_phones)
            tones.extend([0] * len(token_phones))  # 簡易的に全て0とする
            word2ph.append(len(token_phones))  # このトークンが何音素に変換されたか記録
        
        return phones, tones, word2ph
    
    def get_bert_input(self, text: str) -> torch.Tensor:
        """
        テキストをBERTの入力形式に変換
        """
        return self.tokenizer(text, return_tensors="pt")

def create_japanese_bert_tokenizer() -> BertBasedTokenizer:
    """日本語用のBERTトークナイザーを作成"""
    return BertBasedTokenizer("line-corporation/line-distilbert-base-japanese") 
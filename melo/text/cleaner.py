# cleaner.py

from . import chinese, japanese, english, chinese_mix, korean, french, spanish
from . import cleaned_text_to_sequence
import torch
import unicodedata
from sudachipy import tokenizer as sudachi_tokenizer
from sudachipy import dictionary as sudachi_dictionary
import pyopenjtalk
from typing import Tuple, List, Dict
from transformers import AutoTokenizer, AutoModel
import gc

from .symbols import symbols, JP_PHONE_MAPPING

language_module_map = {"ZH": chinese, "JP": japanese, "JA": japanese, "EN": english,
                       'ZH_MIX_EN': chinese_mix, 'KR': korean, 'FR': french, 'SP': spanish, 'ES': spanish}

# グローバルキャッシュ
_bert_model_cache: Dict[str, AutoModel] = {}
_bert_tokenizer_cache = None

def get_bert_model(device: str = "cpu") -> AutoModel:
    """BERTモデルをキャッシュして取得する"""
    global _bert_model_cache
    if device not in _bert_model_cache:
        # GPUメモリ確保のためにキャッシュをクリア
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()
        
        _bert_model_cache[device] = AutoModel.from_pretrained(
            "line-corporation/line-distilbert-base-japanese",
            trust_remote_code=True
        ).to(device)
        
        # 評価モードに設定して最適化
        _bert_model_cache[device].eval()
    
    return _bert_model_cache[device]

def get_bert_tokenizer() -> AutoTokenizer:
    """BERTトークナイザーをキャッシュして取得する"""
    global _bert_tokenizer_cache
    if _bert_tokenizer_cache is None:
        _bert_tokenizer_cache = AutoTokenizer.from_pretrained(
            "line-corporation/line-distilbert-base-japanese",
            trust_remote_code=True
        )
    
    return _bert_tokenizer_cache

def normalize_text(text: str) -> str:
    """テキストの正規化を行う"""
    # 全角文字を半角に変換
    text = unicodedata.normalize('NFKC', text)
    # 特殊文字の置換
    text = text.replace('・', '')
    text = text.replace('ー', 'ー')
    text = text.replace('ッ', 'ッ')
    # 余分な空白を削除
    text = ' '.join(text.split())
    return text

def safe_int_convert(s: str) -> int:
    """
    文字列を整数に変換する。数値でない場合は0を返す
    """
    try:
        return int(s)
    except ValueError:
        return 0

def text_to_phonemes(text: str) -> tuple[list[str], list[int], list[int]]:
    """
    テキストを音素列に変換する
    Args:
        text (str): 入力テキスト
    Returns: (音素リスト, トーンリスト, word2ph)
    """
    norm_text = normalize_text(text)
    
    # japanese.pyのg2pを使用
    phones, tones, word2ph = japanese.g2p(norm_text)
    
    return phones, tones, word2ph

def clean_text(text: str, language: str) -> tuple[str, list[str], list[int], list[int], list[int]]:
    """
    テキストを正規化し、音素情報を抽出する
    Returns: (正規化テキスト, 音素リスト, トーンリスト, word2ph, ph2word)
    """
    if language.upper() in ["JA", "JP"]:
        # 音素変換
        phones, tones, word2ph = text_to_phonemes(text)
        return text, phones, tones, word2ph
    else:
        language_module = language_module_map[language]
        return language_module.clean_text(text)

def get_bert_features(text: str, device: str) -> torch.Tensor:
    """BERT特徴量を取得する"""
    # キャッシュからBERTモデルとトークナイザーを取得
    tokenizer = get_bert_tokenizer()
    model = get_bert_model(device)
    
    # トークン化とBERT特徴量の取得
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        bert = outputs.last_hidden_state.squeeze(0)
        bert = bert.transpose(0, 1)
        return bert

def clean_text_bert(text: str, language: str, device: str = "cpu") -> tuple[str, list[str], list[int], list[int], torch.Tensor]:
    """
    テキストを正規化し、BERT特徴量を取得する
    Returns: (正規化テキスト, 音素リスト, トーンリスト, word2ph, BERT特徴量)
    """
    if language in ["JP", "JA"]:
        norm_text = normalize_text(text)
        # g2p処理で音素リスト、アクセント、word2phを取得
        phones, tones, word2ph = text_to_phonemes(norm_text)
        # BERT特徴量の取得（形状：(embedding_dim, token_length)）
        bert = get_bert_features(norm_text, device)
        
        # BERT特徴量を音素レベルに合わせる
        token_length = bert.shape[1]
        expected_token_entries = len(word2ph)
        phone_length = sum(word2ph)
        
        if token_length == expected_token_entries:
            # 各トークンの特徴量を、word2ph の値だけリピートする
            aligned_features = []
            for i, repeat in enumerate(word2ph):
                token_feature = bert[:, i].unsqueeze(1)  # shape: (embedding_dim, 1)
                aligned_features.append(token_feature.repeat(1, repeat))
            aligned_bert = torch.cat(aligned_features, dim=1)  # shape: (embedding_dim, phone_length)
        else:
            # もし token_length と word2ph の数が一致しない場合は、線形補間でリサイズ
            aligned_bert = align_by_interpolation(bert.transpose(0, 1), phone_length).transpose(0, 1)
        
        return norm_text, phones, tones, word2ph, aligned_bert
    else:
        # 日本語以外の場合は従来の処理
        return clean_text(text, language)

def align_by_interpolation(bert_features, target_length):
    """
    BERTの特徴量をphone長に合わせて補間
    style-bert-vits2方式: BERT×2+1でphoneに合わせる
    """
    device = bert_features.device  # 入力テンソルと同じデバイスを使用
    
    if target_length == bert_features.shape[0] * 2 + 1:
        # BERT×2+1のケース
        output = torch.zeros((target_length, bert_features.shape[1]), device=device)
        for i in range(bert_features.shape[0]):
            # 各BERTトークンを2つの位置にコピー
            output[i*2] = bert_features[i]
            output[i*2+1] = bert_features[i]
        # 最後の位置には最後のBERT特徴量を使用
        output[-1] = bert_features[-1]
        return output
    else:
        # 従来の線形補間（フォールバック）
        source_indices = torch.linspace(0, len(bert_features)-1, target_length, device=device)
        source_indices_floor = source_indices.floor().long()
        source_indices_ceil = source_indices.ceil().long()
        source_indices_frac = source_indices - source_indices_floor

        features_floor = bert_features[source_indices_floor]
        features_ceil = bert_features[source_indices_ceil]

        return features_floor + (features_ceil - features_floor) * source_indices_frac.unsqueeze(-1)

def text_to_sequence(text: str, language: str) -> list[int]:
    """
    テキストをシーケンスに変換する
    """
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

if __name__ == "__main__":
    pass
# japanese_bert.py

import torch
from transformers import AutoModel, AutoTokenizer
from .bert_based_tokenizer import create_japanese_bert_tokenizer
import sys
import numpy as np
import os
import pyopenjtalk
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


# モデルとトークナイザーのキャッシュ
models = {}
tokenizers = {}

# キャッシュ用の辞書
_bert_tokenizer_cache: Dict[str, AutoTokenizer] = {}
_bert_model_cache: Dict[str, AutoModel] = {}

def align_by_interpolation(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    BERT出力（[seq_len, hidden_dim]）を線形補間で target_length に合わせる関数
    
    Args:
        hidden_states: BERTの出力テンソル [seq_len, hidden_dim]
        target_length: 目標の系列長（音素列の長さ）
    
    Returns:
        補間された特徴量テンソル [target_length, hidden_dim]
    """
    # hidden_states: [seq_len, hidden_dim]
    hs = hidden_states.unsqueeze(0).transpose(1, 2)  # shape: [1, hidden_dim, seq_len]
    aligned = F.interpolate(hs, size=target_length, mode='linear', align_corners=False)
    aligned = aligned.transpose(1, 2).squeeze(0)  # shape: [target_length, hidden_dim]
    return aligned

def get_assist_text_by_sentiment(text: str) -> str:
    """
    テキストの感情分析に基づいて補助テキストを生成する関数
    
    Args:
        text: 入力テキスト
    
    Returns:
        感情に応じた補助テキスト
    """
    try:
        import oseti
        analyzer = oseti.Analyzer()
        result = analyzer.analyze(text)
        if result:
            sentiment = result[0]
            if sentiment > 0:
                return "嬉しい"
            elif sentiment < 0:
                return "悲しい"
            else:
                return "普通"
    except Exception as e:
        print(f"感情分析に失敗: {e}")
    return "普通"

def extract_bert_feature_improved(
    text: str,
    word2ph: List[int],
    device: str,
    use_sentiment_assist: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7
) -> torch.Tensor:
    """
    改善されたBERT特徴量抽出関数
    
    Args:
        text: 入力テキスト
        word2ph: トークンごとの音素数のリスト
        device: 使用するデバイス
        use_sentiment_assist: 感情分析による補助テキストを使用するかどうか
        assist_text: カスタム補助テキスト
        assist_text_weight: 補助テキストの重み
    
    Returns:
        BERT特徴量テンソル [音素数, 768]
    """
    try:
        # トークナイザーとモデルの初期化
        model_id = "line-corporation/line-distilbert-base-japanese"
        if model_id not in tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizers[model_id] = tokenizer
        else:
            tokenizer = tokenizers[model_id]

        if model_id not in models:
            model = AutoModel.from_pretrained(model_id, output_hidden_states=True, trust_remote_code=True).to(device)
            models[model_id] = model
        else:
            model = models[model_id]

        # テキストのトークン化
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 補助テキストの処理
        if use_sentiment_assist:
            assist_text = get_assist_text_by_sentiment(text)
        
        if assist_text:
            assist_inputs = tokenizer(assist_text, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                assist_outputs = model(**assist_inputs)
                assist_hidden = assist_outputs.hidden_states[-1][0]  # 最後の層の出力

        # BERT特徴量の抽出
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1][0]  # 最後の層の出力

        # 補助テキストとの重み付き平均
        if assist_text:
            hidden_states = (1 - assist_text_weight) * hidden_states + assist_text_weight * assist_hidden

        # 音素列の長さに合わせて補間
        target_length = sum(word2ph)  # 音素列の総長
        if hidden_states.size(0) != target_length:
            aligned_features = align_by_interpolation(hidden_states, target_length)
        else:
            aligned_features = hidden_states

        return aligned_features

    except Exception as e:
        print(f"BERT特徴量の抽出に失敗: {e}")
        # エラー時は代替特徴量を生成
        return generate_alternative_features(sum(word2ph), device)

def generate_alternative_features(phoneme_count: int, device: str) -> torch.Tensor:
    """
    エラー時の代替特徴量を生成する関数
    
    Args:
        phoneme_count: 音素の数
        device: 使用するデバイス
    
    Returns:
        疑似ランダムな特徴量テンソル [phoneme_count, 768]
    """
    # 疑似ランダムな特徴量を生成（正規分布）
    feature = torch.randn(phoneme_count, 768, device=device)
    # 正規化
    feature = feature / torch.norm(feature, dim=1, keepdim=True)
    return feature

class ImprovedJapaneseBertFeatureExtractor:
    """音素アライメントに基づいた改善されたBERT特徴量抽出器"""
    
    def __init__(self, device=None):
        self.device = device
        if not device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif (
                sys.platform == "darwin"
                and torch.backends.mps.is_available()
            ):
                self.device = "mps"
            else:
                self.device = "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """BERTモデルの初期化"""
        if "line-corporation/line-distilbert-base-japanese" in models:
            self.bert = models["line-corporation/line-distilbert-base-japanese"]
            self.bert_tokenizer = tokenizers["line-corporation/line-distilbert-base-japanese"]
        else:
            try:
                print("モデルをロード中: line-corporation/line-distilbert-base-japanese")
                self.bert = AutoModel.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
                self.bert.eval()
                self.bert = self.bert.to(self.device)
                models["line-corporation/line-distilbert-base-japanese"] = self.bert
                self.bert_tokenizer = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
                tokenizers["line-corporation/line-distilbert-base-japanese"] = self.bert_tokenizer
            except Exception as e:
                print(f" モデ ルのロードに失敗: {e}")
                raise ValueError(f"BERTモデルのロードに失敗しました: {e}")
    
    def get_phoneme_alignment(self, text: str) -> Tuple[List[str], List[float], List[float]]:
        """
        テキストの音素アライメントを計算
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            Tuple[List[str], List[float], List[float]]: (音素リスト, 各トークンの音素開始位置, 各トークンの音素終了位置)
        """
        try:
            phonemes, start_times, end_times = pyopenjtalk.g2p(text, kana=False, join=False)
            return phonemes, start_times, end_times
        except Exception as e:
            print(f"音素アライメントの計算に失敗: {e}")
            return [], [], []
    
    def get_bert_feature(
        self,
        text: str,
        word2ph: List[int],
        use_sentiment_assist: bool = False,
        assist_text: Optional[str] = None,
        assist_text_weight: float = 0.7
    ) -> torch.Tensor:
        """
        BERT特徴量を抽出するメソッド
        
        Args:
            text: 入力テキスト
            word2ph: トークンごとの音素数のリスト
            use_sentiment_assist: 感情分析による補助テキストを使用するかどうか
            assist_text: カスタム補助テキスト
            assist_text_weight: 補助テキストの重み
        
        Returns:
            BERT特徴量テンソル [音素数, 768]
        """
        return extract_bert_feature_improved(
            text,
            word2ph,
            self.device,
            use_sentiment_assist,
            assist_text,
            assist_text_weight
        )

# グローバルな特徴量抽出器のインスタンス
feature_extractor = None

def get_bert_tokenizer() -> AutoTokenizer:
    """
    BERTトークナイザーを取得する
    
    Returns:
        AutoTokenizer: BERTトークナイザー
    """
    if 'tokenizer' not in _bert_tokenizer_cache:
        try:
            _bert_tokenizer_cache['tokenizer'] = AutoTokenizer.from_pretrained(
                "line-corporation/line-distilbert-base-japanese",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"BERTトークナイザーの取得に失敗: {e}")
            raise
    
    return _bert_tokenizer_cache['tokenizer']

def get_bert_model(device: str) -> AutoModel:
    """
    BERTモデルを取得する
    
    Args:
        device: デバイス
    
    Returns:
        AutoModel: BERTモデル
    """
    if 'model' not in _bert_model_cache:
        try:
            model = AutoModel.from_pretrained(
                "line-corporation/line-distilbert-base-japanese",
                trust_remote_code=True
            ).to(device)
            _bert_model_cache['model'] = model
        except Exception as e:
            print(f"BERTモデルの取得に失敗: {e}")
            raise
    
    return _bert_model_cache['model']

def get_bert_feature(text: str, word2ph: List[int], device: str) -> torch.Tensor:
    """
    BERT特徴量を抽出する関数
    
    Args:
        text: 入力テキスト
        word2ph: トークンごとの音素数のリスト
        device: 使用するデバイス
    
    Returns:
        BERT特徴量テンソル [音素数, 768]
    """
    try:
        # トークナイザーとモデルの取得
        tokenizer = get_bert_tokenizer()
        model = get_bert_model(device)
        
        # テキストのトークン化
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # BERT特徴量の抽出
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1][0]  # 最後の層の出力
        
        # word2phに基づいて特徴量を展開
        expanded_features = []
        for i, num_phonemes in enumerate(word2ph):
            # 現在のトークンの特徴量を取得
            token_feature = hidden_states[i]
            # 音素の数だけ繰り返す
            expanded_features.append(token_feature.repeat(num_phonemes, 1))
        
        # すべての特徴量を結合
        return torch.cat(expanded_features, dim=0)
        
    except Exception as e:
        print(f"BERT特徴量の抽出に失敗: {e}")
        # エラー時は代替特徴量を生成
        return generate_alternative_features(sum(word2ph), device)

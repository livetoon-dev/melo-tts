from transformers import AutoTokenizer
import torch

def get_bert_feature(text, word2ph, device=None):
    # LINE DistilBERTを使用した特徴量抽出の実装
    # 既存のコードをそのまま移植
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # トークナイザーとモデルの初期化
    tokenizer = AutoTokenizer.from_pretrained(
        "line-corporation/line-distilbert-base-japanese",
        trust_remote_code=True
    )
    # ... 残りのコードは同じ ... 
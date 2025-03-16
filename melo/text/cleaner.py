from . import chinese, japanese, english, chinese_mix, korean, french, spanish
from . import cleaned_text_to_sequence
import copy
import torch
from typing import Tuple, List

from style_bert_vits2.nlp.japanese.g2p import g2p as g2p_sbv
from style_bert_vits2.nlp.japanese.normalizer import normalize_text as normalize_text_sbv2
from style_bert_vits2.nlp.japanese.bert_feature import (
    extract_bert_feature,
)
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from .symbols import symbols, JP_PHONE_MAPPING
from .bert_based_tokenizer import create_japanese_bert_tokenizer

language_module_map = {"ZH": chinese, "JP": japanese, "JA": japanese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish}

def g2p_pyopenjtalk(text):
    """
    G2P変換をpyopenjtalkまたはpyopenjtalk-plusで行う
    """
    try:
        import pyopenjtalk
    except ImportError:
        raise ImportError("pyopenjtalkまたはpyopenjtalk-plusがインストールされていません。")

    result = pyopenjtalk.g2p(text)
    # pyopenjtalk.g2pが文字列を返す場合、空白で分割する
    if isinstance(result, str):
        phonemes = result.split()
    else:
        phonemes = result

    # 簡易的に全ての音素のトーンは0とする（改善の余地あり）
    tones = [0] * len(phonemes)
    # word2phは、テキストの各文字に対して1音素とし、開始・終了記号を含む
    word2ph = [1] * (len(text) + 2)
    return phonemes, tones, word2ph

def clean_text(text, language, use_sudachi=False):
    language_module = language_module_map[language]
    sbv_norm_text = normalize_text_sbv2(text)

    if use_sudachi:
        print("Sudachiによる分かち書きを使用しています。")
        phones, tones, word2ph = g2p_sbv(sbv_norm_text, use_sudachi=True)
    else:
        print("pyopenjtalk-plus（またはpyopenjtalk）によるG2P変換を使用しています。")
        phones, tones, word2ph = g2p_pyopenjtalk(sbv_norm_text)

    # 音素をフィルタリング
    if language.upper() in ["JA", "JP"]:
        valid_phones = []
        for p in phones:
            if p in symbols:
                valid_phones.append(p)
            elif p in JP_PHONE_MAPPING and JP_PHONE_MAPPING[p] in symbols:
                valid_phones.append(JP_PHONE_MAPPING[p])
            else:
                print(f"Warning: Unknown phone {p} - replacing with '_'")
                valid_phones.append('_')
        phones = valid_phones

    return sbv_norm_text, phones, tones, word2ph


def clean_text_bert(text: str, language: str, device=None, use_sudachi=False) -> Tuple[str, List[str], List[int], List[int], torch.Tensor]:
    """
    テキストを正規化し、BERT特徴量と音素情報を抽出する
    
    Args:
        text (str): 入力テキスト
        language (str): 言語コード
        device: 計算デバイス
        use_sudachi (bool): 使用しない（互換性のために残している）
    
    Returns:
        Tuple[str, List[str], List[int], List[int], torch.Tensor]:
            - 正規化されたテキスト
            - 音素列
            - トーン情報
            - word2ph（各トークンの音素数）
            - BERT特徴量
    """
    if language.upper() in ["JA", "JP"]:
        # BERTトークナイザーベースの処理を使用
        tokenizer = create_japanese_bert_tokenizer()
        
        # テキストの正規化とG2P変換
        phones, tones, word2ph = tokenizer.process_text(text)
        
        # 音素のフィルタリングと標準化
        valid_phones = []
        for phone in phones:
            if phone in symbols:
                valid_phones.append(phone)
            elif phone in JP_PHONE_MAPPING and JP_PHONE_MAPPING[phone] in symbols:
                valid_phones.append(JP_PHONE_MAPPING[phone])
            else:
                print(f"Warning: Unknown phone {phone} - replacing with '_'")
                valid_phones.append('_')
        
        # BERT特徴量の抽出
        from .japanese_bert import get_bert_feature
        bert = get_bert_feature(text, word2ph, device=device)
        
        return normalize_text_sbv2(text), valid_phones, tones, word2ph, bert
    else:
        # 他の言語の処理（変更なし）
        language_module = language_module_map[language]
        return language_module.clean_text_bert(text)


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
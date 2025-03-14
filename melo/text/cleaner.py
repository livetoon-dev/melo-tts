from . import chinese, japanese, english, chinese_mix, korean, french, spanish
from . import cleaned_text_to_sequence
import copy
import torch

from style_bert_vits2.nlp.japanese.g2p import g2p as g2p_sbv
from style_bert_vits2.nlp.japanese.normalizer import normalize_text as normalize_text_sbv2
from style_bert_vits2.nlp.japanese.bert_feature import (
    extract_bert_feature,
)
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from .symbols import symbols

language_module_map = {"ZH": chinese, "JP": japanese, "JA": japanese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish}

# 日本語の音素をmeloのシンボルと対応させるためのマッピング
JP_PHONE_MAPPING = {
    'h': 'h',
    'u': 'u',
    't': 't',
    'n': 'n',
    'y': 'y',
    'q': 'q',
    'w': 'w',
    'N': 'N',
    "'": "'"
}

def clean_text(text, language, use_sudachi=False):
    language_module = language_module_map[language]
    sbv_norm_text = normalize_text_sbv2(text)
    phones, tones, word2ph = g2p_sbv(sbv_norm_text, use_sudachi=use_sudachi)
    
    # 音素をフィルタリング
    if language == "JA" or language == "JP":
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


def clean_text_bert(text, language, device=None, use_sudachi=False):
    try:
        # 日本語の場合、BERTモデルを事前に読み込む
        if language == "JA" or language == "JP":
            # クリーン処理
            sbv_norm_text = normalize_text_sbv2(text)
            print(f"Processing with sbv2: {sbv_norm_text}")
            
            # G2P処理 - 例外が発生しても処理を継続
            try:
                phones, tones, word2ph = g2p_sbv(sbv_norm_text, use_sudachi=use_sudachi)
                print(f"G2P result - phones: {phones}, tones: {tones}, word2ph: {word2ph}")
            except Exception as e:
                print(f"G2P processing error: {str(e)}")
                # ダミーデータを生成
                phones = ["_"] + ["a"] * len(sbv_norm_text) + ["_"]  # 単純な代替音素列
                tones = [0] * len(phones)
                word2ph = [1] * (len(sbv_norm_text) + 2)
            
            # 音素をフィルタリング 
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
            
            # BERTモデルがまだ読み込まれていない場合は読み込む
            try:
                # Hugging Faceから直接モデルを読み込む
                bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
                bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
                # BERT特徴量抽出
                bert = extract_bert_feature(sbv_norm_text, word2ph, device=device)
            except Exception as e:
                print(f"BERT feature extraction failed: {str(e)}")
                print("Using dummy BERT features instead")
                # ダミーのBERT特徴量を生成
                bert_dim = 1024
                bert_length = len(phones)
                bert = torch.zeros((bert_dim, bert_length)).to(device if device else "cpu")
            
            return sbv_norm_text, phones, tones, word2ph, bert
        else:
            # その他の言語
            language_module = language_module_map[language]
            sbv_norm_text = normalize_text_sbv2(text)
            phones, tones, word2ph = g2p_sbv(sbv_norm_text, use_sudachi=use_sudachi)
            
            # ダミーのBERT特徴量を生成
            bert_dim = 1024
            bert_length = len(phones)
            bert = torch.zeros((bert_dim, bert_length)).to(device if device else "cpu")
            
            return sbv_norm_text, phones, tones, word2ph, bert
    except Exception as e:
        print(f"Error in clean_text_bert: {str(e)}")
        raise


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
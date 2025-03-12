from . import chinese, japanese, english, chinese_mix, korean, french, spanish
from . import cleaned_text_to_sequence
import copy

from style_bert_vits2.nlp.japanese.g2p import g2p as g2p_sbv
from style_bert_vits2.nlp.japanese.normalizer import normalize_text as normalize_text_sbv2
from style_bert_vits2.nlp.japanese.bert_feature import (
    extract_bert_feature,
)
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish}


def clean_text(text, language):
    language_module = language_module_map[language]
    sbv_norm_text = normalize_text_sbv2(text)
    phones, tones, word2ph = g2p_sbv(sbv_norm_text)
    return sbv_norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = language_module_map[language]
    sbv_norm_text = normalize_text_sbv2(text)
    phones, tones, word2ph = g2p_sbv(sbv_norm_text)
    
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = extract_bert_feature(sbv_norm_text, word2ph, device=device)
    
    return sbv_norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
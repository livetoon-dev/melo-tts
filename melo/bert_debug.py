#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = 'true'

"""
BERT特徴量と音素列の不一致の原因をデバッグするスクリプト
"""

import sys
import torch
from text.bert_based_tokenizer import create_japanese_bert_tokenizer
from text.cleaner import clean_text_bert
from text.symbols import symbols, JP_PHONE_MAPPING

def debug_bert_phoneme_mismatch(text: str, device: str = 'cuda'):
    """
    BERT特徴量と音素列の不一致をデバッグする
    """
    print(f"\n=== デバッグ開始 ===")
    print(f"入力テキスト: '{text}'")
    print(f"使用デバイス: {device}")
    
    # BERTトークナイザーの初期化
    print("\n1. BERTトークナイザーの初期化")
    tokenizer = create_japanese_bert_tokenizer()
    
    # 2. テキストの正規化
    norm_text = tokenizer.normalize_text(text)
    print(f"\n2. テキストの正規化:")
    print(f"正規化前: '{text}'")
    print(f"正規化後: '{norm_text}'")
    
    # 3. BERTトークナイザーによる分かち書き
    tokens = tokenizer.tokenizer(norm_text, return_tensors="pt")
    token_ids = tokens["input_ids"][0]
    token_strs = tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\n3. BERTトークナイザーによる分かち書き:")
    print(f"トークン: {token_strs}")
    print(f"トークン数: {len(token_strs)}")
    
    # 4. トークンごとのG2P変換
    print(f"\n4. G2P変換処理:")
    phones, tones, word2ph = tokenizer.process_text(norm_text)
    print(f"音素列: {phones}")
    print(f"トーン: {tones}")
    print(f"word2ph: {word2ph}")
    print(f"音素数: {len(phones)}")
    print(f"トーン数: {len(tones)}")
    print(f"word2ph長: {len(word2ph)}")
    print(f"word2ph合計: {sum(word2ph)}")
    
    # 5. 音素のフィルタリング
    print(f"\n5. 音素のフィルタリング:")
    valid_phones = []
    for phone in phones:
        if phone in symbols:
            valid_phones.append(phone)
            print(f"音素 '{phone}' はそのまま使用")
        elif phone in JP_PHONE_MAPPING and JP_PHONE_MAPPING[phone] in symbols:
            mapped = JP_PHONE_MAPPING[phone]
            valid_phones.append(mapped)
            print(f"音素 '{phone}' を '{mapped}' に変換")
        else:
            valid_phones.append('_')
            print(f"警告: 未知の音素 '{phone}' を '_' に置換")
    
    print(f"\n変換後の音素列: {valid_phones}")
    
    # 6. clean_text_bert関数による処理
    print(f"\n6. clean_text_bert関数の実行:")
    try:
        norm_text, final_phones, final_tones, final_word2ph, bert = clean_text_bert(text, "JA", device=device)
        print(f"正規化テキスト: '{norm_text}'")
        print(f"最終音素列: {final_phones}")
        print(f"最終トーン: {final_tones}")
        print(f"最終word2ph: {final_word2ph}")
        print(f"BERT特徴量の形状: {bert.shape}")
        
        # 7. 最終的な長さの整合性チェック
        print(f"\n7. 最終結果の整合性チェック:")
        print(f"音素数: {len(final_phones)}")
        print(f"トーン数: {len(final_tones)}")
        print(f"BERT特徴量の長さ: {bert.shape[1]}")
        
        if len(final_phones) == len(final_tones) == bert.shape[1]:
            print("✓ すべての長さが一致しています")
        else:
            print("✗ 長さの不一致があります:")
            print(f"  音素数: {len(final_phones)}")
            print(f"  トーン数: {len(final_tones)}")
            print(f"  BERT特徴量の長さ: {bert.shape[1]}")
            
        # 8. トークンと音素の対応関係の表示
        print(f"\n8. トークンと音素の対応関係:")
        token_idx = 0
        phone_idx = 0
        for token, ph_len in zip(token_strs, final_word2ph):
            phones_for_token = final_phones[phone_idx:phone_idx + ph_len]
            print(f"トークン '{token}' → 音素 {phones_for_token} (長さ: {ph_len})")
            token_idx += 1
            phone_idx += ph_len
        
    except Exception as e:
        print(f"エラーが発生: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # テスト対象のテキスト
    test_text = "これはテストです。" if len(sys.argv) < 2 else sys.argv[1]
    
    # GPUが利用可能ならGPU、そうでなければCPUを使用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    
    # デバッグ実行
    debug_bert_phoneme_mismatch(test_text, device)

if __name__ == "__main__":
    main() 
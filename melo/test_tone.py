import pyopenjtalk
from text.japanese import text_to_sep_kata

def safe_int_convert(s: str) -> int:
    """
    文字列を整数に変換する。数値でない場合は0を返す
    """
    try:
        return int(s)
    except ValueError:
        return 0

def test_phoneme_tone(text: str):
    """
    テキストから音素とトーンを取得して表示するテスト関数
    """
    print(f"\n=== テスト: {text} ===")
    
    # カタカナに変換
    words, katas = text_to_sep_kata(text)
    
    # 音素とトーンを取得
    all_phones = []
    all_tones = []
    word2ph = []
    
    for word, kata in zip(words, katas):
        try:
            # pyopenjtalkを使用して音素変換
            phonemes = pyopenjtalk.g2p(kata)
            if isinstance(phonemes, str):
                phonemes = phonemes.split()
            
            if phonemes:
                # フルコンテキストラベルを取得してアクセント情報を解析
                full_context = pyopenjtalk.extract_fullcontext(kata)
                
                # アクセント情報を解析
                word_tones = []
                for label in full_context:
                    # A:フィールドからアクセント情報を抽出
                    if 'A:' in label:
                        acc_info = label.split('A:')[1].split('/')[0]
                        acc_parts = acc_info.split('+')
                        
                        # アクセント情報を安全に取得
                        acc_dist = safe_int_convert(acc_parts[0])
                        mora_pos = safe_int_convert(acc_parts[1])
                        mora_count = safe_int_convert(acc_parts[2])
                        
                        # アクセント核の判定（xxの場合は0として扱う）
                        if acc_dist == 0:  # アクセント核
                            word_tones.append(1)
                        elif acc_dist > 0:  # アクセント核以降
                            word_tones.append(0)
                        else:  # アクセント核以前
                            word_tones.append(1)
                
                # 音素とトーンを追加
                all_phones.extend(phonemes)
                all_tones.extend(word_tones)
                word2ph.append(len(phonemes))
            else:
                # 音素変換が完全に失敗した場合
                all_phones.append('_')
                all_tones.append(0)
                word2ph.append(1)
        except Exception as e:
            print(f"音素変換に失敗: {word} (読み: {kata}): {e}")
            all_phones.append('_')
            all_tones.append(0)
            word2ph.append(1)
    
    # train.list形式で出力
    print(f"dummy.wav|vtuber|JP|{text}|{' '.join(all_phones)}|{' '.join(map(str, all_tones))}|{' '.join(map(str, word2ph))}")

if __name__ == "__main__":
    # テストケース
    test_cases = [
        "こんにちは",
        "ありがとう",
        "おはよう",
        "さようなら",
        "こんばんは"
    ]
    
    for text in test_cases:
        test_phoneme_tone(text)

import pyopenjtalk
from text.japanese import g2p
from text.symbols import ja_symbols, punctuation, pad

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
    
    # g2p関数を使用して音素とトーンを取得
    phones, tones, word2ph = g2p(text)
    
    # 音素とトーンの対応を詳細表示
    print("音素-トーン対応:")
    for p, t in zip(phones, tones):
        print(f"{p}: {t}", end=" ")
    print("\n")
    
    # train.list形式で出力
    print(f"dummy.wav|vtuber|JP|{text}|{' '.join(phones)}|{' '.join(map(str, tones))}|{' '.join(map(str, word2ph))}")

if __name__ == "__main__":
    # テストケース
    test_cases = [
        "こんにちは",  # ko↑Nnichiwa
        "ありがとう",  # a↑rigato:
        "おはよう",    # o↑hayo:
        "本当",        # ho↑Nto:
        "私",          # wata↑shi
        "東京",        # to:↑kyo:
        "こんにちは！",
        "おはよう？",
        "すごい！！",
        "本当ですか？？",
    ]
    
    print("=== 音素・トーンテスト ===")
    print(f"利用可能な音素: {sorted(ja_symbols)}")
    print(f"利用可能な句読点: {sorted(punctuation)}")
    print(f"パッド記号: {pad}\n")
    
    for text in test_cases:
        test_phoneme_tone(text)

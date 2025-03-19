import unittest
from .cleaner import clean_text_bert

class TestCleanTextBert(unittest.TestCase):
    def test_basic_case(self):
        """基本的なケースのテスト"""
        text = "こんにちは"
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, "JP")
        
        print("\n=== 基本的なケース ===")
        print(f"入力テキスト: {text}")
        print(f"正規化テキスト: {norm_text}")
        print(f"音素: {' '.join(phones)}")
        print(f"トーン: {' '.join(map(str, tones))}")
        print(f"word2ph: {' '.join(map(str, word2ph))}")
        
        # 音素とword2phの長さが一致することを確認
        self.assertEqual(len(phones), sum(word2ph))
        # 音素とトーンの長さが一致することを確認
        self.assertEqual(len(phones), len(tones))

    def test_sokuon_case(self):
        """促音を含むケースのテスト"""
        text = "やっば"
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, "JP")
        
        print("\n=== 促音を含むケース ===")
        print(f"入力テキスト: {text}")
        print(f"正規化テキスト: {norm_text}")
        print(f"音素: {' '.join(phones)}")
        print(f"トーン: {' '.join(map(str, tones))}")
        print(f"word2ph: {' '.join(map(str, word2ph))}")
        
        # 促音が正しく処理されているか確認
        self.assertIn('Q', phones)  # 促音記号が含まれているか
        self.assertEqual(len(phones), sum(word2ph))

    def test_special_phonemes(self):
        """特殊な音素を含むケースのテスト"""
        text = "んんん"
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, "JP")
        
        print("\n=== 特殊な音素を含むケース ===")
        print(f"入力テキスト: {text}")
        print(f"正規化テキスト: {norm_text}")
        print(f"音素: {' '.join(phones)}")
        print(f"トーン: {' '.join(map(str, tones))}")
        print(f"word2ph: {' '.join(map(str, word2ph))}")
        
        # 特殊な音素が正しく処理されているか確認
        self.assertTrue(all(ph in ['N', 'n'] for ph in phones))
        self.assertEqual(len(phones), sum(word2ph))

    def test_error_case(self):
        """エラーが発生しそうなケースのテスト"""
        text = "漢字漢字漢字"  # 漢字のみのケース
        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, "JP")
        
        print("\n=== エラーが発生しそうなケース ===")
        print(f"入力テキスト: {text}")
        print(f"正規化テキスト: {norm_text}")
        print(f"音素: {' '.join(phones)}")
        print(f"トーン: {' '.join(map(str, tones))}")
        print(f"word2ph: {' '.join(map(str, word2ph))}")
        
        # エラーが発生せずに処理されているか確認
        self.assertEqual(len(phones), sum(word2ph))
        self.assertEqual(len(phones), len(tones))

if __name__ == '__main__':
    unittest.main() 
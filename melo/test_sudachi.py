from sudachipy import tokenizer
from sudachipy import dictionary
import unicodedata

def test_sudachi():
    # トークナイザーの初期化
    tokenizer_obj = dictionary.Dictionary().create(mode=tokenizer.Tokenizer.SplitMode.C)
    
    # テストケース
    test_cases = [
        "なんか年々冬って寒くなってる気がしない?",
        "ちょっとびっくりしただけかー。喜んでもらえたならよかった",
        "食べさせて",
        "あはは、ほんと何落ち込んでんだろうね",
        "動画もいいんですけど、たまにはお友達や恋人さんとお外に出かけるのもいいと思いますよ",
        "じゃあこの絵渡しちゃったら、一緒に衣装探しに行こう",
        "なかったことにしようよ、2人で",
        "人の頭上を自転車で飛び越ちておいて、言いたいことはそれだけか"
    ]
    
    for text in test_cases:
        print(f"\nテストテキスト: {text}")
        print("-" * 50)
        
        # テキストの正規化
        text = unicodedata.normalize('NFKC', text)
        
        # トークン化
        tokens = tokenizer_obj.tokenize(text)
        
        # 各トークンの情報を表示
        for token in tokens:
            surface = token.surface()
            reading = token.reading_form()
            pos = token.part_of_speech()
            print(f"表層形: {surface}")
            print(f"読み: {reading}")
            print(f"品詞: {pos}")
            print("-" * 30)

if __name__ == "__main__":
    test_sudachi() 
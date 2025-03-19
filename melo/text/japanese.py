# japanese.py

# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata
from typing import List, Tuple
from sudachipy import tokenizer as sudachi_tokenizer
from sudachipy import dictionary as sudachi_dictionary
import pyopenjtalk

from transformers import AutoTokenizer

from .symbols import ja_symbols, punctuation

# MeCabのインポートをコメントアウトして、例外を回避
# try:
#     import MeCab
# except ImportError as e:
#     raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e
from num2words import num2words

# Sudachi初期化は一度だけ
_sudachi_dict = sudachi_dictionary.Dictionary(dict="full")
_sudachi_tokenizer_obj = _sudachi_dict.create(mode=sudachi_tokenizer.Tokenizer.SplitMode.C)

# 句読点の定義
PUNCTUATIONS = [',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '-', '—', '–', '…']

# 母音の定義
VOWELS = ['a', 'i', 'u', 'e', 'o']

# カタカナから音素へのマッピング
MORA_KATA_TO_MORA_PHONEMES = {
    'ア': (None, 'a'), 'イ': (None, 'i'), 'ウ': (None, 'u'), 'エ': (None, 'e'), 'オ': (None, 'o'),
    'カ': ('k', 'a'), 'キ': ('k', 'i'), 'ク': ('k', 'u'), 'ケ': ('k', 'e'), 'コ': ('k', 'o'),
    'サ': ('s', 'a'), 'シ': ('sh', 'i'), 'ス': ('s', 'u'), 'セ': ('s', 'e'), 'ソ': ('s', 'o'),
    'タ': ('t', 'a'), 'チ': ('ch', 'i'), 'ツ': ('ts', 'u'), 'テ': ('t', 'e'), 'ト': ('t', 'o'),
    'ナ': ('n', 'a'), 'ニ': ('n', 'i'), 'ヌ': ('n', 'u'), 'ネ': ('n', 'e'), 'ノ': ('n', 'o'),
    'ハ': ('h', 'a'), 'ヒ': ('h', 'i'), 'フ': ('f', 'u'), 'ヘ': ('h', 'e'), 'ホ': ('h', 'o'),
    'マ': ('m', 'a'), 'ミ': ('m', 'i'), 'ム': ('m', 'u'), 'メ': ('m', 'e'), 'モ': ('m', 'o'),
    'ヤ': ('y', 'a'), 'ユ': ('y', 'u'), 'ヨ': ('y', 'o'),
    'ラ': ('r', 'a'), 'リ': ('r', 'i'), 'ル': ('r', 'u'), 'レ': ('r', 'e'), 'ロ': ('r', 'o'),
    'ワ': ('w', 'a'), 'ヲ': ('w', 'o'), 'ン': ('N', None),
    'ガ': ('g', 'a'), 'ギ': ('g', 'i'), 'グ': ('g', 'u'), 'ゲ': ('g', 'e'), 'ゴ': ('g', 'o'),
    'ザ': ('z', 'a'), 'ジ': ('j', 'i'), 'ズ': ('z', 'u'), 'ゼ': ('z', 'e'), 'ゾ': ('z', 'o'),
    'ダ': ('d', 'a'), 'ヂ': ('j', 'i'), 'ヅ': ('z', 'u'), 'デ': ('d', 'e'), 'ド': ('d', 'o'),
    'バ': ('b', 'a'), 'ビ': ('b', 'i'), 'ブ': ('b', 'u'), 'ベ': ('b', 'e'), 'ボ': ('b', 'o'),
    'パ': ('p', 'a'), 'ピ': ('p', 'i'), 'プ': ('p', 'u'), 'ペ': ('p', 'e'), 'ポ': ('p', 'o'),
    'キャ': ('ky', 'a'), 'キュ': ('ky', 'u'), 'キョ': ('ky', 'o'),
    'シャ': ('sh', 'a'), 'シュ': ('sh', 'u'), 'ショ': ('sh', 'o'),
    'チャ': ('ch', 'a'), 'チュ': ('ch', 'u'), 'チョ': ('ch', 'o'),
    'ニャ': ('ny', 'a'), 'ニュ': ('ny', 'u'), 'ニョ': ('ny', 'o'),
    'ヒャ': ('hy', 'a'), 'ヒュ': ('hy', 'u'), 'ヒョ': ('hy', 'o'),
    'ミャ': ('my', 'a'), 'ミュ': ('my', 'u'), 'ミョ': ('my', 'o'),
    'リャ': ('ry', 'a'), 'リュ': ('ry', 'u'), 'リョ': ('ry', 'o'),
    'ギャ': ('gy', 'a'), 'ギュ': ('gy', 'u'), 'ギョ': ('gy', 'o'),
    'ジャ': ('j', 'a'), 'ジュ': ('j', 'u'), 'ジョ': ('j', 'o'),
    'ビャ': ('by', 'a'), 'ビュ': ('by', 'u'), 'ビョ': ('by', 'o'),
    'ピャ': ('py', 'a'), 'ピュ': ('py', 'u'), 'ピョ': ('py', 'o'),
}

def text_to_sep_kata(text: str, raise_yomi_error: bool = False) -> tuple[list[str], list[str]]:
    """
    テキストをカタカナに変換する
    Args:
        text (str): 正規化されたテキスト
        raise_yomi_error (bool): Falseの場合、読めない文字が「'」として発音される
    Returns: (単語リスト, カタカナリスト)
    """
    # Sudachiでトークン化
    tokens = _sudachi_tokenizer_obj.tokenize(text)
    
    words = []
    katas = []
    
    for token in tokens:
        word = token.surface()
        yomi = token.reading_form()
        
        if not yomi:
            if raise_yomi_error:
                raise ValueError(f"読めない文字があります: {word}")
            else:
                yomi = "'"  # 読めない文字の場合は「'」を使用
        
        words.append(word)
        katas.append(yomi)
    
    return words, katas

def text_normalize(text):
    """日本語テキストの正規化を行う関数"""
    # 全角文字を半角に変換
    text = unicodedata.normalize('NFKC', text)
    
    # 数字を漢数字に変換
    text = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='ja'), text)
    
    # 特殊文字の置換
    text = text.replace('ー', 'ー')
    text = text.replace('～', 'ー')
    text = text.replace('…', '...')
    text = text.replace('—', '-')
    text = text.replace('―', '-')
    text = text.replace('－', '-')
    text = text.replace('‐', '-')
    text = text.replace('‑', '-')
    text = text.replace('−', '-')
    text = text.replace('「', '"')
    text = text.replace('」', '"')
    text = text.replace('『', '"')
    text = text.replace('』', '"')
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('［', '[')
    text = text.replace('］', ']')
    text = text.replace('【', '[')
    text = text.replace('】', ']')
    text = text.replace('〔', '{')
    text = text.replace('〕', '}')
    text = text.replace('《', '<')
    text = text.replace('》', '>')
    text = text.replace('〈', '<')
    text = text.replace('〉', '>')
    text = text.replace('《', '<')
    text = text.replace('》', '>')
    text = text.replace('『', '"')
    text = text.replace('』', '"')
    text = text.replace('「', '"')
    text = text.replace('」', '"')
    text = text.replace('・', ' ')
    text = text.replace('…', '...')
    text = text.replace('〜', '~')
    text = text.replace('～', '~')
    text = text.replace('ー', 'ー')
    text = text.replace('―', '-')
    text = text.replace('－', '-')
    text = text.replace('‐', '-')
    text = text.replace('‑', '-')
    text = text.replace('−', '-')
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('［', '[')
    text = text.replace('］', ']')
    text = text.replace('【', '[')
    text = text.replace('】', ']')
    text = text.replace('〔', '{')
    text = text.replace('〕', '}')
    text = text.replace('《', '<')
    text = text.replace('》', '>')
    text = text.replace('〈', '<')
    text = text.replace('〉', '>')
    text = text.replace('《', '<')
    text = text.replace('》', '>')
    text = text.replace('『', '"')
    text = text.replace('』', '"')
    text = text.replace('「', '"')
    text = text.replace('」', '"')
    text = text.replace('・', ' ')
    text = text.replace('…', '...')
    text = text.replace('〜', '~')
    text = text.replace('～', '~')
    
    return text

_CONVRULES = [
    # Conversion of 2 letters
    "アァ/ a a",
    "イィ/ i i",
    "イェ/ i e",
    "イャ/ y a",
    "ウゥ/ u:",
    "エェ/ e e",
    "オォ/ o:",
    "カァ/ k a:",
    "キィ/ k i:",
    "クゥ/ k u:",
    "クャ/ ky a",
    "クュ/ ky u",
    "クョ/ ky o",
    "ケェ/ k e:",
    "コォ/ k o:",
    "ガァ/ g a:",
    "ギィ/ g i:",
    "グゥ/ g u:",
    "グャ/ gy a",
    "グュ/ gy u",
    "グョ/ gy o",
    "ゲェ/ g e:",
    "ゴォ/ g o:",
    "サァ/ s a:",
    "シィ/ sh i:",
    "スゥ/ s u:",
    "スャ/ sh a",
    "スュ/ sh u",
    "スョ/ sh o",
    "セェ/ s e:",
    "ソォ/ s o:",
    "ザァ/ z a:",
    "ジィ/ j i:",
    "ズゥ/ z u:",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ゼェ/ z e:",
    "ゾォ/ z o:",
    "タァ/ t a:",
    "チィ/ ch i:",
    "ツァ/ ts a",
    "ツィ/ ts i",
    "ツゥ/ ts u:",
    "ツャ/ ch a",
    "ツュ/ ch u",
    "ツョ/ ch o",
    "ツェ/ ts e",
    "ツォ/ ts o",
    "テェ/ t e:",
    "トォ/ t o:",
    "ダァ/ d a:",
    "ヂィ/ j i:",
    "ヅゥ/ d u:",
    "ヅャ/ zy a",
    "ヅュ/ zy u",
    "ヅョ/ zy o",
    "デェ/ d e:",
    "ドォ/ d o:",
    "ナァ/ n a:",
    "ニィ/ n i:",
    "ヌゥ/ n u:",
    "ヌャ/ ny a",
    "ヌュ/ ny u",
    "ヌョ/ ny o",
    "ネェ/ n e:",
    "ノォ/ n o:",
    "ハァ/ h a:",
    "ヒィ/ h i:",
    "フゥ/ f u:",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "ヘェ/ h e:",
    "ホォ/ h o:",
    "バァ/ b a:",
    "ビィ/ b i:",
    "ブゥ/ b u:",
    "フャ/ hy a",
    "ブュ/ by u",
    "フョ/ hy o",
    "ベェ/ b e:",
    "ボォ/ b o:",
    "パァ/ p a:",
    "ピィ/ p i:",
    "プゥ/ p u:",
    "プャ/ py a",
    "プュ/ py u",
    "プョ/ py o",
    "ペェ/ p e:",
    "ポォ/ p o:",
    "マァ/ m a:",
    "ミィ/ m i:",
    "ムゥ/ m u:",
    "ムャ/ my a",
    "ムュ/ my u",
    "ムョ/ my o",
    "メェ/ m e:",
    "モォ/ m o:",
    "ヤァ/ y a:",
    "ユゥ/ y u:",
    "ユャ/ y a:",
    "ユュ/ y u:",
    "ユョ/ y o:",
    "ヨォ/ y o:",
    "ラァ/ r a:",
    "リィ/ r i:",
    "ルゥ/ r u:",
    "ルャ/ ry a",
    "ルュ/ ry u",
    "ルョ/ ry o",
    "レェ/ r e:",
    "ロォ/ r o:",
    "ワァ/ w a:",
    "ヲォ/ o:",
    "ディ/ d i",
    "デェ/ d e:",
    "デャ/ dy a",
    "デュ/ dy u",
    "デョ/ dy o",
    "ティ/ t i",
    "テェ/ t e:",
    "テャ/ ty a",
    "テュ/ ty u",
    "テョ/ ty o",
    "スィ/ s i",
    "ズァ/ z u a",
    "ズィ/ z i",
    "ズゥ/ z u",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ズェ/ z e",
    "ズォ/ z o",
    "キャ/ ky a",
    "キュ/ ky u",
    "キョ/ ky o",
    "シャ/ sh a",
    "シュ/ sh u",
    "シェ/ sh e",
    "ショ/ sh o",
    "チャ/ ch a",
    "チュ/ ch u",
    "チェ/ ch e",
    "チョ/ ch o",
    "トゥ/ t u",
    "トャ/ ty a",
    "トュ/ ty u",
    "トョ/ ty o",
    "ドァ/ d o a",
    "ドゥ/ d u",
    "ドャ/ dy a",
    "ドュ/ dy u",
    "ドョ/ dy o",
    "ドォ/ d o:",
    "ニャ/ ny a",
    "ニュ/ ny u",
    "ニョ/ ny o",
    "ヒャ/ hy a",
    "ヒュ/ hy u",
    "ヒョ/ hy o",
    "ミャ/ my a",
    "ミュ/ my u",
    "ミョ/ my o",
    "リャ/ ry a",
    "リュ/ ry u",
    "リョ/ ry o",
    "ギャ/ gy a",
    "ギュ/ gy u",
    "ギョ/ gy o",
    "ヂェ/ j e",
    "ヂャ/ j a",
    "ヂュ/ j u",
    "ヂョ/ j o",
    "ジェ/ j e",
    "ジャ/ j a",
    "ジュ/ j u",
    "ジョ/ j o",
    "ビャ/ by a",
    "ビュ/ by u",
    "ビョ/ by o",
    "ピャ/ py a",
    "ピュ/ py u",
    "ピョ/ py o",
    "ウァ/ u a",
    "ウィ/ w i",
    "ウェ/ w e",
    "ウォ/ w o",
    "ファ/ f a",
    "フィ/ f i",
    "フゥ/ f u",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "フェ/ f e",
    "フォ/ f o",
    "ヴァ/ b a",
    "ヴィ/ b i",
    "ヴェ/ b e",
    "ヴォ/ b o",
    "ヴュ/ by u",
    # Conversion of 1 letter
    "ア/ a",
    "イ/ i",
    "ウ/ u",
    "エ/ e",
    "オ/ o",
    "カ/ k a",
    "キ/ k i",
    "ク/ k u",
    "ケ/ k e",
    "コ/ k o",
    "サ/ s a",
    "シ/ sh i",
    "ス/ s u",
    "セ/ s e",
    "ソ/ s o",
    "タ/ t a",
    "チ/ ch i",
    "ツ/ ts u",
    "テ/ t e",
    "ト/ t o",
    "ナ/ n a",
    "ニ/ n i",
    "ヌ/ n u",
    "ネ/ n e",
    "ノ/ n o",
    "ハ/ h a",
    "ヒ/ h i",
    "フ/ f u",
    "ヘ/ h e",
    "ホ/ h o",
    "マ/ m a",
    "ミ/ m i",
    "ム/ m u",
    "メ/ m e",
    "モ/ m o",
    "ラ/ r a",
    "リ/ r i",
    "ル/ r u",
    "レ/ r e",
    "ロ/ r o",
    "ガ/ g a",
    "ギ/ g i",
    "グ/ g u",
    "ゲ/ g e",
    "ゴ/ g o",
    "ザ/ z a",
    "ジ/ j i",
    "ズ/ z u",
    "ゼ/ z e",
    "ゾ/ z o",
    "ダ/ d a",
    "ヂ/ j i",
    "ヅ/ z u",
    "デ/ d e",
    "ド/ d o",
    "バ/ b a",
    "ビ/ b i",
    "ブ/ b u",
    "ベ/ b e",
    "ボ/ b o",
    "パ/ p a",
    "ピ/ p i",
    "プ/ p u",
    "ペ/ p e",
    "ポ/ p o",
    "ヤ/ y a",
    "ユ/ y u",
    "ヨ/ y o",
    "ワ/ w a",
    "ヰ/ i",
    "ヱ/ e",
    "ヲ/ o",
    "ン/ N",
    "ッ/ q",
    "ヴ/ b u",
    "ー/:",
    # Try converting broken text
    "ァ/ a",
    "ィ/ i",
    "ゥ/ u",
    "ェ/ e",
    "ォ/ o",
    "ヮ/ w a",
    "ォ/ o",
    # Try converting broken text
    "ャ/ y a",
    "ョ/ y o",
    "ュ/ y u",
    "琦/ ch i",
    "ヶ/ k e",
    "髙/ t a k a",
    "煞/ sh y a",
    # Symbols
    "、/ ,",
    "。/ .",
    "！/ !",
    "？/ ?",
    "・/ ,",
]

_COLON_RX = re.compile(":+")
_REJECT_RX = re.compile("[^ a-zA-Z:,.?]")


def _makerulemap():
    l = [tuple(x.split("/")) for x in _CONVRULES]
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))


_RULEMAP1, _RULEMAP2 = _makerulemap()


# ja_symbolsを使いやすいリストに変換
ja_symbols_list = ja_symbols

# 日本語の音素変換に必要な定数
# トークナイザーのインポート
model_id = 'ku-nlp/deberta-v2-base-japanese-char-wwm'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 音素分配関数
def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

# 利用可能な音素を確認（ja_symbolsリストから）
valid_phonemes = set(ja_symbols)
valid_phonemes.update(punctuation)  # 句読点も有効な音素として追加

# print("利用可能な音素リスト:", sorted(list(valid_phonemes)))

# 文字から音素へのマッピング
# 必ず ja_symbols に含まれる音素のみを使用する
char_to_phonemes = {
    # 基本母音
    'あ': ['a'],    'い': ['i'],    'う': ['u'],    'え': ['e'],    'お': ['o'],
    'ア': ['a'],    'イ': ['i'],    'ウ': ['u'],    'エ': ['e'],    'オ': ['o'],
    'ぁ': ['a'],    'ぃ': ['i'],    'ぅ': ['u'],    'ぇ': ['e'],    'ぉ': ['o'],
    'ァ': ['a'],    'ィ': ['i'],    'ゥ': ['u'],    'ェ': ['e'],    'ォ': ['o'],
    
    # か行
    'か': ['k', 'a'], 'き': ['k', 'i'], 'く': ['k', 'u'], 'け': ['k', 'e'], 'こ': ['k', 'o'],
    'カ': ['k', 'a'], 'キ': ['k', 'i'], 'ク': ['k', 'u'], 'ケ': ['k', 'e'], 'コ': ['k', 'o'],
    # が行
    'が': ['g', 'a'], 'ぎ': ['g', 'i'], 'ぐ': ['g', 'u'], 'げ': ['g', 'e'], 'ご': ['g', 'o'],
    'ガ': ['g', 'a'], 'ギ': ['g', 'i'], 'グ': ['g', 'u'], 'ゲ': ['g', 'e'], 'ゴ': ['g', 'o'],
    # さ行
    'さ': ['s', 'a'], 'す': ['s', 'u'], 'せ': ['s', 'e'], 'そ': ['s', 'o'],
    'サ': ['s', 'a'], 'ス': ['s', 'u'], 'セ': ['s', 'e'], 'ソ': ['s', 'o'],
    'し': ['sh', 'i'], 'シ': ['sh', 'i'],
    # ざ行
    'ざ': ['z', 'a'], 'ず': ['z', 'u'], 'ぜ': ['z', 'e'], 'ぞ': ['z', 'o'],
    'ザ': ['z', 'a'], 'ズ': ['z', 'u'], 'ゼ': ['z', 'e'], 'ゾ': ['z', 'o'],
    'じ': ['j', 'i'], 'ジ': ['j', 'i'], 'ぢ': ['j', 'i'], 'ヂ': ['j', 'i'],
    'づ': ['z', 'u'], 'ヅ': ['z', 'u'],
    # た行
    'た': ['t', 'a'], 'て': ['t', 'e'], 'と': ['t', 'o'],
    'タ': ['t', 'a'], 'テ': ['t', 'e'], 'ト': ['t', 'o'],
    'ち': ['ch', 'i'], 'チ': ['ch', 'i'],
    'つ': ['ts', 'u'], 'ツ': ['ts', 'u'],
    # だ行
    'だ': ['d', 'a'], 'で': ['d', 'e'], 'ど': ['d', 'o'],
    'ダ': ['d', 'a'], 'デ': ['d', 'e'], 'ド': ['d', 'o'],
    # な行
    'な': ['n', 'a'], 'に': ['n', 'i'], 'ぬ': ['n', 'u'], 'ね': ['n', 'e'], 'の': ['n', 'o'],
    'ナ': ['n', 'a'], 'ニ': ['n', 'i'], 'ヌ': ['n', 'u'], 'ネ': ['n', 'e'], 'ノ': ['n', 'o'],
    # は行
    'は': ['h', 'a'], 'ひ': ['h', 'i'], 'へ': ['h', 'e'], 'ほ': ['h', 'o'],
    'ハ': ['h', 'a'], 'ヒ': ['h', 'i'], 'ヘ': ['h', 'e'], 'ホ': ['h', 'o'],
    'ふ': ['f', 'u'], 'フ': ['f', 'u'],
    # ば行
    'ば': ['b', 'a'], 'び': ['b', 'i'], 'ぶ': ['b', 'u'], 'べ': ['b', 'e'], 'ぼ': ['b', 'o'],
    'バ': ['b', 'a'], 'ビ': ['b', 'i'], 'ブ': ['b', 'u'], 'ベ': ['b', 'e'], 'ボ': ['b', 'o'],
    # ぱ行
    'ぱ': ['p', 'a'], 'ぴ': ['p', 'i'], 'ぷ': ['p', 'u'], 'ぺ': ['p', 'e'], 'ぽ': ['p', 'o'],
    'パ': ['p', 'a'], 'ピ': ['p', 'i'], 'プ': ['p', 'u'], 'ペ': ['p', 'e'], 'ポ': ['p', 'o'],
    # ま行
    'ま': ['m', 'a'], 'み': ['m', 'i'], 'む': ['m', 'u'], 'め': ['m', 'e'], 'も': ['m', 'o'],
    'マ': ['m', 'a'], 'ミ': ['m', 'i'], 'ム': ['m', 'u'], 'メ': ['m', 'e'], 'モ': ['m', 'o'],
    # や行
    'や': ['y', 'a'], 'ゆ': ['y', 'u'], 'よ': ['y', 'o'],
    'ヤ': ['y', 'a'], 'ユ': ['y', 'u'], 'ヨ': ['y', 'o'],
    'ゃ': ['y', 'a'], 'ゅ': ['y', 'u'], 'ょ': ['y', 'o'],
    'ャ': ['y', 'a'], 'ュ': ['y', 'u'], 'ョ': ['y', 'o'],
    # ら行
    'ら': ['r', 'a'], 'り': ['r', 'i'], 'る': ['r', 'u'], 'れ': ['r', 'e'], 'ろ': ['r', 'o'],
    'ラ': ['r', 'a'], 'リ': ['r', 'i'], 'ル': ['r', 'u'], 'レ': ['r', 'e'], 'ロ': ['r', 'o'],
    # わ行
    'わ': ['w', 'a'], 'を': ['o'], 'ん': ['N'],
    'ワ': ['w', 'a'], 'ヲ': ['o'], 'ン': ['N'],
    'ゎ': ['w', 'a'], 'ヮ': ['w', 'a'],
    
    # 拗音（きゃ、しゃなど）
    'きゃ': ['ky', 'a'], 'きゅ': ['ky', 'u'], 'きょ': ['ky', 'o'],
    'キャ': ['ky', 'a'], 'キュ': ['ky', 'u'], 'キョ': ['ky', 'o'],
    'ぎゃ': ['gy', 'a'], 'ぎゅ': ['gy', 'u'], 'ぎょ': ['gy', 'o'],
    'ギャ': ['gy', 'a'], 'ギュ': ['gy', 'u'], 'ギョ': ['gy', 'o'],
    'しゃ': ['sh', 'a'], 'しゅ': ['sh', 'u'], 'しょ': ['sh', 'o'],
    'シャ': ['sh', 'a'], 'シュ': ['sh', 'u'], 'ショ': ['sh', 'o'],
    'じゃ': ['j', 'a'], 'じゅ': ['j', 'u'], 'じょ': ['j', 'o'],
    'ジャ': ['j', 'a'], 'ジュ': ['j', 'u'], 'ジョ': ['j', 'o'],
    'ちゃ': ['ch', 'a'], 'ちゅ': ['ch', 'u'], 'ちょ': ['ch', 'o'],
    'チャ': ['ch', 'a'], 'チュ': ['ch', 'u'], 'チョ': ['ch', 'o'],
    'にゃ': ['ny', 'a'], 'にゅ': ['ny', 'u'], 'にょ': ['ny', 'o'],
    'ニャ': ['ny', 'a'], 'ニュ': ['ny', 'u'], 'ニョ': ['ny', 'o'],
    'ひゃ': ['hy', 'a'], 'ひゅ': ['hy', 'u'], 'ひょ': ['hy', 'o'],
    'ヒャ': ['hy', 'a'], 'ヒュ': ['hy', 'u'], 'ヒョ': ['hy', 'o'],
    'びゃ': ['by', 'a'], 'びゅ': ['by', 'u'], 'びょ': ['by', 'o'],
    'ビャ': ['by', 'a'], 'ビュ': ['by', 'u'], 'ビョ': ['by', 'o'],
    'ぴゃ': ['py', 'a'], 'ぴゅ': ['py', 'u'], 'ぴょ': ['py', 'o'],
    'ピャ': ['py', 'a'], 'ピュ': ['py', 'u'], 'ピョ': ['py', 'o'],
    'みゃ': ['my', 'a'], 'みゅ': ['my', 'u'], 'みょ': ['my', 'o'],
    'ミャ': ['my', 'a'], 'ミュ': ['my', 'u'], 'ミョ': ['my', 'o'],
    'りゃ': ['ry', 'a'], 'りゅ': ['ry', 'u'], 'りょ': ['ry', 'o'],
    'リャ': ['ry', 'a'], 'リュ': ['ry', 'u'], 'リョ': ['ry', 'o'],
    
    # 促音
    'っ': ['q'], 'ッ': ['q'],
    
    # 長音
    'ー': [':'],
    
    # 記号類
    '　': ['SP'], ' ': ['SP'],   # スペース
    '、': [','], '。': ['.'],   # 句読点
    '？': ['?'], '！': ['!'],   # 疑問符と感嘆符
    '…': ['...'],             # 省略記号
}

# 漢字→カタカナ変換テーブル
kanji_to_kana = {
    # 数字
    '一': 'イチ', '二': 'ニ', '三': 'サン', '四': 'ヨン', '五': 'ゴ',
    '六': 'ロク', '七': 'ナナ', '八': 'ハチ', '九': 'キュウ', '十': 'ジュウ',
    '百': 'ヒャク', '千': 'セン', '万': 'マン', '億': 'オク',
    # 代名詞
    '私': 'ワタシ', '僕': 'ボク', '俺': 'オレ', '君': 'キミ', '彼': 'カレ', '彼女': 'カノジョ',
    # 方向・位置
    '上': 'ウエ', '下': 'シタ', '左': 'ヒダリ', '右': 'ミギ', '前': 'マエ', '後': 'アト', '横': 'ヨコ',
    '東': 'ヒガシ', '西': 'ニシ', '南': 'ミナミ', '北': 'キタ',
    # 自然
    '山': 'ヤマ', '川': 'カワ', '海': 'ウミ', '空': 'ソラ', '地': 'チ', '森': 'モリ',
    '雨': 'アメ', '雪': 'ユキ', '風': 'カゼ', '火': 'ヒ', '水': 'ミズ', '木': 'キ', '草': 'クサ',
    # 時間
    '時': 'ジ', '分': 'フン', '秒': 'ビョウ', '間': 'カン', '今': 'イマ', '昨': 'サク', '明': 'メイ',
    '年': 'ネン', '月': 'ツキ', '日': 'ヒ', '曜': 'ヨウ', '週': 'シュウ',
    '朝': 'アサ', '昼': 'ヒル', '夜': 'ヨル', '夕': 'ユウ', '晩': 'バン',
    '春': 'ハル', '夏': 'ナツ', '秋': 'アキ', '冬': 'フユ',
    # 色
    '赤': 'アカ', '青': 'アオ', '黄': 'キ', '緑': 'ミドリ', '白': 'シロ', '黒': 'クロ', '色': 'イロ',
    # 感覚
    '音': 'オト', '声': 'コエ', '力': 'チカラ', '歌': 'ウタ', '色': 'イロ',
    # 動詞関連
    '見': 'ミ', '聞': 'キ', '話': 'ハナシ', '読': 'ヨ', '書': 'カ', '言': 'イ',
    '食': 'タ', '飲': 'ノ', '寝': 'ネ', '起': 'オ', '歩': 'アル',
    '走': 'ハシ', '飛': 'ト', '泳': 'オヨ', '笑': 'ワラ', '泣': 'ナ',
    '好': 'ス', '嫌': 'キラ', '楽': 'タノ', '苦': 'クル', '嬉': 'ウレ',
    '悲': 'カナ', '怒': 'オコ', '驚': 'オドロ', '恐': 'コワ',
    # 場所・建物
    '家': 'イエ', '学': 'ガク', '校': 'コウ', '社': 'シャ', '寺': 'テラ',
    '店': 'ミセ', '道': 'ミチ', '駅': 'エキ', '橋': 'ハシ',
    # その他よく使う漢字
    '人': 'ヒト', '子': 'コ', '女': 'オンナ', '男': 'オトコ', '犬': 'イヌ', '猫': 'ネコ',
    '大': 'ダイ', '小': 'ショウ', '中': 'チュウ', '新': 'シン', '古': 'フル',
    '多': 'オオ', '少': 'スコ', '高': 'タカ', '低': 'ヒク', '長': 'ナガ', '短': 'ミジカ',
    '行': 'イ', '来': 'キ', '帰': 'カエ', '入': 'イ', '出': 'デ',
    '開': 'ア', '閉': 'シ', '始': 'ハジ', '終': 'オ', '続': 'ツズ',
    '作': 'ツク', '壊': 'コワ', '直': 'ナオ', '変': 'カ', '決': 'キ',
    '知': 'シ', '思': 'オモ', '考': 'カンガ', '信': 'シン', '頑': 'ガン',
    '張': 'バ', '全': 'ゼン', '部': 'ブ', '半': 'ハン', '分': 'ブン',
    '次': 'ツギ', '回': 'カイ', '度': 'ド', '数': 'カズ', '実': 'ジツ', '伝': 'デン', 
    '本': 'ホン', '当': 'トウ', '物': 'モノ', '事': 'コト', '心': 'ココロ', '手': 'テ', '足': 'アシ',
    '目': 'メ', '耳': 'ミミ', '口': 'クチ', '顔': 'カオ', '頭': 'アタマ',
}

# 基本単語マッピング（2文字以上のよく使う単語）
word_to_kana = {
    '次回': 'ジカイ',
    '実際': 'ジッサイ',
    '羊': 'ヒツジ',
    'こんにちは': 'コンニチハ',
    'さようなら': 'サヨウナラ',
    '本当': 'ホントウ',
    '今日': 'キョウ',
    '明日': 'アシタ',
    '昨日': 'キノウ',
    'ありがとう': 'アリガトウ',
    'すみません': 'スミマセン',
    'こんばんは': 'コンバンハ',
    'おはよう': 'オハヨウ',
    '大丈夫': 'ダイジョウブ',
}

# 音素変換関数をまとめて提供
def text_to_phonemes(text):
    """
    日本語テキストを音素に変換します。
    ja_symbolsに含まれている音素のみを使用します。
    """
    if not text or text.isspace():
        return []
    
    # まず単語単位で変換を試行
    for word, kana in word_to_kana.items():
        if word in text:
            text = text.replace(word, kana)
    
    # 次に漢字をカタカナに変換
    for kanji, kana in kanji_to_kana.items():
        if kanji in text:
            text = text.replace(kanji, kana)
    
    # 文字を音素に変換
    phonemes = []
    i = 0
    while i < len(text):
        # まず2文字の組み合わせをチェック（拗音対応）
        if i < len(text) - 1 and text[i:i+2] in char_to_phonemes:
            char_phones = char_to_phonemes[text[i:i+2]]
            phonemes.extend(char_phones)
            i += 2
        # 次に1文字をチェック
        elif text[i] in char_to_phonemes:
            char_phones = char_to_phonemes[text[i]]
            phonemes.extend(char_phones)
            i += 1
        # 句読点などの記号
        elif text[i] in punctuation:
            phonemes.append(text[i])
            i += 1
        # それ以外は未知の文字
        else:
            print(f"注意: 未知の文字 '{text[i]}' をスキップします")
            i += 1
    
    # 音素が有効かどうか確認
    valid_phones = []
    for ph in phonemes:
        if ph in valid_phonemes:
            valid_phones.append(ph)
        else:
            print(f"注意: 音素 '{ph}' はja_symbolsに含まれていないため削除します")
    
    return valid_phones

def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    """
    テキストを音素列に変換する
    Returns: (音素リスト, トーンリスト, word2ph)
    """
    # テキストを正規化
    norm_text = text_normalize(text)
    
    # 単語とカタカナに分割
    words, katas = text_to_sep_kata(norm_text)
    
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
                        acc_dist = int(acc_parts[0]) if acc_parts[0].isdigit() else 0
                        mora_pos = int(acc_parts[1]) if acc_parts[1].isdigit() else 0
                        mora_count = int(acc_parts[2]) if acc_parts[2].isdigit() else 0
                        
                        # アクセント核の判定
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
    
    return all_phones, all_tones, word2ph

def get_bert_feature(text, word2ph, device):
    from text import japanese_bert
    return japanese_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    text = "こんにちは、世界！..."
    text = 'ええ、僕はおきなと申します。こちらの小さいわらべは杏子。ご挨拶が遅れてしまいすみません。あなたの名は?'
    text = 'あの、お前以外のみんなは、全員生きてること?'
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)

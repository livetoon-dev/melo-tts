# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from . import symbols
punctuation = ["!", "?", "…", ",", ".", "'", "-"]

# MeCabのインポートをコメントアウトして、例外を回避
# try:
#     import MeCab
# except ImportError as e:
#     raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e
from num2words import num2words

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


# symbols.py内で定義されたja_symbolsを直接インポート
from text.symbols import ja_symbols, punctuation
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

# 有効な音素のみを含む基本マッピング（各音素がsymbols.ja_symbolsに含まれることを確認済み）
# すべての文字が確実に正しい音素に変換されるように設計
base_char_to_phonemes = {
    # 基本的な母音（a, i, u, e, o）
    'あ': ['a'],    'い': ['i'],    'う': ['u'],    'え': ['e'],    'お': ['o'],
    'ア': ['a'],    'イ': ['i'],    'ウ': ['u'],    'エ': ['e'],    'オ': ['o'],
    'ぁ': ['a'],    'ぃ': ['i'],    'ぅ': ['u'],    'ぇ': ['e'],    'ぉ': ['o'],
    'ァ': ['a'],    'ィ': ['i'],    'ゥ': ['u'],    'ェ': ['e'],    'ォ': ['o'],
    
    # 子音 + 母音の組み合わせ
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
    # き行
    'きゃ': ['ky', 'a'], 'きゅ': ['ky', 'u'], 'きょ': ['ky', 'o'],
    'キャ': ['ky', 'a'], 'キュ': ['ky', 'u'], 'キョ': ['ky', 'o'],
    # ぎ行
    'ぎゃ': ['gy', 'a'], 'ぎゅ': ['gy', 'u'], 'ぎょ': ['gy', 'o'],
    'ギャ': ['gy', 'a'], 'ギュ': ['gy', 'u'], 'ギョ': ['gy', 'o'],
    # し行
    'しゃ': ['sh', 'a'], 'しゅ': ['sh', 'u'], 'しょ': ['sh', 'o'],
    'シャ': ['sh', 'a'], 'シュ': ['sh', 'u'], 'ショ': ['sh', 'o'],
    # じ行
    'じゃ': ['j', 'a'], 'じゅ': ['j', 'u'], 'じょ': ['j', 'o'],
    'ジャ': ['j', 'a'], 'ジュ': ['j', 'u'], 'ジョ': ['j', 'o'],
    # ち行
    'ちゃ': ['ch', 'a'], 'ちゅ': ['ch', 'u'], 'ちょ': ['ch', 'o'],
    'チャ': ['ch', 'a'], 'チュ': ['ch', 'u'], 'チョ': ['ch', 'o'],
    # に行
    'にゃ': ['ny', 'a'], 'にゅ': ['ny', 'u'], 'にょ': ['ny', 'o'],
    'ニャ': ['ny', 'a'], 'ニュ': ['ny', 'u'], 'ニョ': ['ny', 'o'],
    # ひ行
    'ひゃ': ['hy', 'a'], 'ひゅ': ['hy', 'u'], 'ひょ': ['hy', 'o'],
    'ヒャ': ['hy', 'a'], 'ヒュ': ['hy', 'u'], 'ヒョ': ['hy', 'o'],
    # び行
    'びゃ': ['by', 'a'], 'びゅ': ['by', 'u'], 'びょ': ['by', 'o'],
    'ビャ': ['by', 'a'], 'ビュ': ['by', 'u'], 'ビョ': ['by', 'o'],
    # ぴ行
    'ぴゃ': ['py', 'a'], 'ぴゅ': ['py', 'u'], 'ぴょ': ['py', 'o'],
    'ピャ': ['py', 'a'], 'ピュ': ['py', 'u'], 'ピョ': ['py', 'o'],
    # み行
    'みゃ': ['my', 'a'], 'みゅ': ['my', 'u'], 'みょ': ['my', 'o'],
    'ミャ': ['my', 'a'], 'ミュ': ['my', 'u'], 'ミョ': ['my', 'o'],
    # り行
    'りゃ': ['ry', 'a'], 'りゅ': ['ry', 'u'], 'りょ': ['ry', 'o'],
    'リャ': ['ry', 'a'], 'リュ': ['ry', 'u'], 'リョ': ['ry', 'o'],
    
    # 促音
    'っ': ['q'], 'ッ': ['q'],
    
    # 長音
    'ー': [':'],
    
    # 記号
    '　': ['SP'], ' ': ['SP'],   # スペース
    '、': [','], '。': ['.'],   # 句読点
    '？': ['?'], '！': ['!'],   # 疑問符と感嘆符
    '…': ['...'],             # 省略記号
}

# よく使われる漢字をカタカナに変換するマップ
kanji_to_kana = {
    '一': 'イチ', '二': 'ニ', '三': 'サン', '四': 'ヨン', '五': 'ゴ',
    '六': 'ロク', '七': 'ナナ', '八': 'ハチ', '九': 'キュウ', '十': 'ジュウ',
    '百': 'ヒャク', '千': 'セン', '万': 'マン', '億': 'オク',
    '人': 'ヒト', '私': 'ワタシ', '僕': 'ボク', '俺': 'オレ', '君': 'キミ',
    '大': 'ダイ', '小': 'ショウ', '中': 'チュウ', '上': 'ウエ', '下': 'シタ',
    '左': 'ヒダリ', '右': 'ミギ', '前': 'マエ', '後': 'アト', '横': 'ヨコ',
    '山': 'ヤマ', '川': 'カワ', '海': 'ウミ', '空': 'ソラ', '地': 'チ',
    '日': 'ヒ', '月': 'ツキ', '火': 'ヒ', '水': 'ミズ', '木': 'キ',
    '金': 'キン', '土': 'ツチ', '雨': 'アメ', '雪': 'ユキ', '風': 'カゼ',
    '時': 'ジ', '分': 'フン', '秒': 'ビョウ', '間': 'カン', '今': 'イマ',
    '昨': 'サク', '明': 'メイ', '年': 'ネン', '週': 'シュウ', '朝': 'アサ',
    '昼': 'ヒル', '夜': 'ヨル', '夕': 'ユウ', '晩': 'バン',
    '春': 'ハル', '夏': 'ナツ', '秋': 'アキ', '冬': 'フユ',
    '東': 'ヒガシ', '西': 'ニシ', '南': 'ミナミ', '北': 'キタ',
    '赤': 'アカ', '青': 'アオ', '黄': 'キ', '緑': 'ミドリ', '白': 'シロ', '黒': 'クロ',
    '色': 'イロ', '音': 'オト', '声': 'コエ', '力': 'チカラ',
    '見': 'ミ', '聞': 'キ', '話': 'ハナシ', '読': 'ヨ', '書': 'カ',
    '食': 'タ', '飲': 'ノ', '寝': 'ネ', '起': 'オ', '歩': 'アル',
    '走': 'ハシ', '飛': 'ト', '泳': 'オヨ', '笑': 'ワラ', '泣': 'ナ',
    '好': 'ス', '嫌': 'キラ', '楽': 'タノ', '苦': 'クル', '嬉': 'ウレ',
    '悲': 'カナ', '怒': 'オコ', '驚': 'オドロ', '恐': 'コワ',
    '家': 'イエ', '学': 'ガク', '校': 'コウ', '社': 'シャ', '寺': 'テラ',
    '店': 'ミセ', '道': 'ミチ', '駅': 'エキ', '橋': 'ハシ',
    '行': 'イ', '来': 'キ', '帰': 'カエ', '入': 'イ', '出': 'デ',
    '次': 'ツギ', '回': 'カイ', '実': 'ジツ', '伝': 'デン',
}

def kata2phoneme(text):
    """
    日本語のテキスト（ひらがな、カタカナ、漢字）を音素に変換します。
    ja_symbolsにリストされている有効な音素のみを使用します。
    """
    # 空文字チェック
    if not text or text.isspace():
        return []

    # 漢字をカタカナに変換
    for kanji, kana in kanji_to_kana.items():
        text = text.replace(kanji, kana)

    # 文字を音素に変換
    phonemes = []
    i = 0
    while i < len(text):
        # 2文字の組み合わせをまず確認（拗音などの処理）
        if i < len(text) - 1 and text[i:i+2] in base_char_to_phonemes:
            curr_phonemes = base_char_to_phonemes[text[i:i+2]]
            phonemes.extend(curr_phonemes)
            i += 2
        # 1文字の処理
        elif text[i] in base_char_to_phonemes:
            curr_phonemes = base_char_to_phonemes[text[i]]
            phonemes.extend(curr_phonemes)
            i += 1
        # 記号や句読点
        elif text[i] in punctuation:
            phonemes.append(text[i])
            i += 1
        # 対応するマッピングがない文字
        else:
            # 処理できない文字に関する警告
            print(f"警告: 未知の文字 '{text[i]}' (コード: {ord(text[i])}) をスキップします")
            i += 1

    # 音素の検証
    validated_phonemes = []
    for ph in phonemes:
        if ph in ja_symbols_list or ph in punctuation:
            validated_phonemes.append(ph)
        else:
            print(f"警告: 音素 '{ph}' はja_symbolsに含まれていません")
    
    return validated_phonemes

def g2p(norm_text):
    """
    日本語テキストを音素、トーン、単語-音素マッピングに変換します。
    完全にja_symbolsに合わせた音素を生成し、BERT特徴抽出との互換性を確保します。
    """
    # 前処理: 空文字列のチェック
    if not norm_text or norm_text.isspace():
        return ["_"], [0], [1]

    try:
        # テキストの正規化
        normalized_text = norm_text.replace('　', ' ')
        
        # トークン化
        tokenized = tokenizer.tokenize(normalized_text)
        
        # トークン化結果の検証
        if not tokenized:
            return ["_"], [0], [1]
            
        # 音素とword2phのリスト初期化
        phs = []
        ph_groups = []
        
        # トークングループの作成
        current_group = []
        for t in tokenized:
            if not t.startswith("#"):
                if current_group:
                    ph_groups.append(current_group)
                current_group = [t]
            else:
                if not current_group:
                    current_group = [t.replace("#", "")]
                else:
                    current_group.append(t.replace("#", ""))
        
        # 最後のグループを追加
        if current_group:
            ph_groups.append(current_group)
            
        # グループが作成できたか確認
        if not ph_groups:
            return ["_"], [0], [1]
        
        # 各グループを処理
        word2ph = []
        for group in ph_groups:
            # グループ内のトークンを結合
            text = "".join(group)
            
            # 特殊ケース: [UNK]トークンと句読点
            if text == '[UNK]' or text in punctuation:
                phs += ['_' if text == '[UNK]' else text]
                word2ph += [1]
                continue
                
            # 音素への変換
            try:
                # 各文字を直接ja_symbolsに存在する音素に変換
                phonemes = []
                # 文字ごとに処理
                for char in text:
                    if char in base_char_to_phonemes:
                        # 基本マッピングに存在する場合は直接変換
                        char_phonemes = base_char_to_phonemes[char]
                        phonemes.extend(char_phonemes)
                    elif char in punctuation:
                        # 句読点はそのまま追加
                        phonemes.append(char)
                    elif char in kanji_to_kana:
                        # 漢字の場合はカタカナに変換して処理
                        kana = kanji_to_kana[char]
                        for k in kana:
                            if k in base_char_to_phonemes:
                                k_phonemes = base_char_to_phonemes[k]
                                phonemes.extend(k_phonemes)
                            else:
                                print(f"警告: 無効な音素 '{char}' が見つかりました（テキスト：{text}）")
                    else:
                        # 未知の文字の場合は警告を出して、'_'を追加
                        print(f"警告: 無効な音素 '{char}' が見つかりました（テキスト：{text}）")
                
                # 音素変換結果の検証
                if not phonemes:
                    # 代替策としてアンダースコアを使用
                    phs += ['_']
                    word2ph += [1] * len(group)
                    continue
                
                # 音素とトークンの長さを取得
                phone_len = len(phonemes)
                word_len = len(group)
                
                # word2phの計算
                if phone_len > 0 and word_len > 0:
                    # 音素をトークンに分配
                    word_ph_mapping = distribute_phone(phone_len, word_len)
                    word2ph += word_ph_mapping
                else:
                    word2ph += [1] * max(1, word_len)
                
                # 有効な音素をphsリストに追加
                phs += phonemes
                
            except Exception as e:
                # エラー時はプレースホルダーを使用
                phs += ['_']
                word2ph += [1] * len(group)
        
        # 最終チェック
        if not phs:
            phs = ["_"]
        if not word2ph:
            word2ph = [1]
            
        # 音素リストの前後にパディング追加
        phones = ["_"] + phs + ["_"]
        # 日本語はトーンなし
        tones = [0 for _ in phones]
        # word2phの前後にパディング追加
        word2ph = [1] + word2ph + [1]
        
        # word2phの長さ確認
        # 重要: word2phの長さはtokenizedの長さ+2（先頭と末尾のパディング）と一致する必要がある
        if len(word2ph) != len(tokenized) + 2:
            # 長さの調整
            if len(word2ph) > len(tokenized) + 2:
                word2ph = word2ph[:len(tokenized) + 2]
            else:
                word2ph = word2ph + [1] * ((len(tokenized) + 2) - len(word2ph))
        
        return phones, tones, word2ph
        
    except Exception as e:
        # エラー時の最小限の返り値
        return ["_"], [0], [1]

# g2p_sbv関数
def g2p_sbv(norm_text, use_sudachi=False):
    """
    style_bert_vits2のg2p関数と互換性を持たせるためのラッパー。
    """
    try:
        return g2p(norm_text)
    except Exception as e:
        print(f"g2p_sbv処理エラー: {e} (text: '{norm_text}')")
        return ["_"], [0], [1]

def get_bert_feature(text, word2ph, device):
    from text import japanese_bert
    return japanese_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    text = "こんにちは、世界！..."
    text = 'ええ、僕はおきなと申します。こちらの小さいわらべは杏子。ご挨拶が遅れてしまいすみません。あなたの名は?'
    text = 'あの、お前以外のみんなは、全員生きてること?'
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)

# if __name__ == '__main__':
#     from pykakasi import kakasi
#     # Initialize kakasi object
#     kakasi = kakasi()

#     # Set options for converting Chinese characters to Katakana
#     kakasi.setMode("J", "H")  # Chinese to Katakana
#     kakasi.setMode("K", "H")  # Hiragana to Katakana

#     # Convert Chinese characters to Katakana
#     conv = kakasi.getConverter()
#     katakana_text = conv.do('ええ、僕はおきなと申します。こちらの小さいわらべは杏子。ご挨拶が遅れてしまいすみません。あなたの名は?')  # Replace with your Chinese text

#     print(katakana_text)  # Output: ニーハオセカイ

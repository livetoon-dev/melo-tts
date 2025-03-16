from typing import Optional

import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    language: Languages = Languages.JP,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    日本語のテキストから BERT の特徴量を抽出する

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        device (str): 推論に利用するデバイス
        language (Languages, optional): 言語 (デフォルト: Languages.JP)
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # languageがNoneの場合はJPを使用
    if language is None:
        language = Languages.JP
        
    try:
        model = bert_models.load_model(language).to(device)  # type: ignore
    except Exception as e:
        print(f"モデルのロードエラー: {e}")
        raise e

    style_res_mean = None
    with torch.no_grad():
        try:
            tokenizer = bert_models.load_tokenizer(language, trust_remote_code=True)
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)  # type: ignore
            
            # LINE DistilBERTモデルの場合はoutput_hidden_statesを指定
            res = model(**inputs, output_hidden_states=True)
            
            # DistilBERTは6層なので最終3層のみを取得
            if len(res["hidden_states"]) <= 6:  # DistilBERTの場合
                res = torch.cat(res["hidden_states"][-2:-1], -1)[0].cpu()
            else:  # 従来のDeBERTaの場合
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
                
            if assist_text:
                style_inputs = tokenizer(assist_text, return_tensors="pt")
                for i in style_inputs:
                    style_inputs[i] = style_inputs[i].to(device)  # type: ignore
                style_res = model(**style_inputs, output_hidden_states=True)
                
                # DistilBERTの場合の対応
                if len(style_res["hidden_states"]) <= 6:
                    style_res = torch.cat(style_res["hidden_states"][-2:-1], -1)[0].cpu()
                else:
                    style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
                    
                style_res_mean = style_res.mean(0)
        except Exception as e:
            print(f"BERT特徴量抽出処理エラー: {str(e)}")
            print(f"テキスト: '{text}'")
            print(f"word2ph: {word2ph}")
            print(f"assist_text: {assist_text}")
            print(f"tokenizer type: {type(tokenizer).__name__}")
            print(f"model type: {type(model).__name__}")
            import traceback
            traceback.print_exc()
            raise

    try:
        # テキスト長+2（特殊トークン含む）とword2ph長の不一致チェック
        if len(word2ph) != len(text) + 2:
            print(f"警告: word2phの長さ({len(word2ph)})がテキスト長+2({len(text) + 2})と一致しません。テキスト: '{text}'")
            # テキスト長に合わせてword2phを調整
            if len(word2ph) > len(text) + 2:
                # word2phが長すぎる場合は切り詰める
                word2phone = word2ph[:len(text) + 2]
                print(f"word2phを短縮: {word2phone}")
            else:
                # word2phが短すぎる場合は拡張する
                word2phone = word2ph + [1] * ((len(text) + 2) - len(word2ph))
                print(f"word2phを拡張: {word2phone}")
        else:
            word2phone = word2ph
        
        # トークン数とword2phの長さの不一致を検出して調整
        if len(res) != len(word2phone):
            print(f"警告: トークン数({len(res)})とword2ph長({len(word2phone)})が一致しません。調整します。")
            
            # トークン数に合わせてword2phを調整（BERTの方が優先）
            if len(res) < len(word2phone):
                # word2phを短くする - BERTトークン数を優先
                word2phone = word2phone[:len(res)]
                print(f"word2phを短縮: {word2phone}")
            else:
                # word2phを延長（各要素に1を追加）
                word2phone = word2phone + [1] * (len(res) - len(word2phone))
                print(f"word2phを延長: {word2phone}")
        
        # 特徴量の生成
        phone_level_feature = []
        expected_total_phones = sum(word2phone)  # 期待される音素の総数
        
        for i in range(len(word2phone)):
            if i >= len(res):
                print(f"警告: インデックス {i} が範囲外です (res size: {len(res)})")
                break
            
            # 現在のトークンの特徴量を取得して音素数分繰り返す
            if assist_text:
                assert style_res_mean is not None
                repeat_feature = (
                    res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                    + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
                )
            else:
                repeat_feature = res[i].repeat(word2phone[i], 1)
            
            phone_level_feature.append(repeat_feature)

        # 音素レベルの特徴量を結合
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        
        # 実際の音素数とword2phから計算された期待値を比較
        actual_phones = phone_level_feature.shape[0]
        if actual_phones != expected_total_phones:
            print(f"警告: 実際の音素数({actual_phones})と期待される音素数({expected_total_phones})が一致しません")
            
            # 音素数を調整（足りない場合は最後の特徴量を繰り返し、多すぎる場合は切り詰める）
            if actual_phones < expected_total_phones:
                # 不足分を最後の特徴量で埋める
                last_feature = phone_level_feature[-1].unsqueeze(0)
                missing = expected_total_phones - actual_phones
                extension = last_feature.repeat(missing, 1)
                phone_level_feature = torch.cat([phone_level_feature, extension], dim=0)
                print(f"音素レベル特徴量を拡張: {actual_phones} → {expected_total_phones}")
            else:
                # 余分な特徴量を切り捨てる
                phone_level_feature = phone_level_feature[:expected_total_phones]
                print(f"音素レベル特徴量を短縮: {actual_phones} → {expected_total_phones}")
        
        # 最終的な特徴量の形状を確認して返す
        print(f"最終特徴量の形状: {phone_level_feature.T.shape}")
        return phone_level_feature.T

    except Exception as e:
        print(f"特徴量変換エラー: {str(e)}")
        print(f"テキスト: '{text}'")
        print(f"word2ph: {word2ph}")
        print(f"res shape: {res.shape}")
        import traceback
        traceback.print_exc()
        raise

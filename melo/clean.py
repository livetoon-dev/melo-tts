import collections
import re

input_file = "/home/ydk/workspace/melo-tts/melo/data/metadata_0308_melo.list"
output_file = "/home/ydk/workspace/melo-tts/melo/data/metadata_filtered.list"

filtered_lines = []

# 특수 문자 및 기호를 제외하는 정규식 패턴
pattern = re.compile(r"[^a-zA-Zぁ-んァ-ン一-龥0-9]")  # 영어, 일본어(히라가나, 가타카나, 한자), 숫자만 남김

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue  # 잘못된 형식 무시
        text = parts[3]  # | 뒤에 있는 텍스트

        # 기호 제거 후 문자만 남기기
        cleaned_text = pattern.sub("", text)

        # 문자 빈도 계산
        counter = collections.Counter(cleaned_text)

        # 13번 이상 반복되는 문자 포함 여부 확인
        if any(freq >= 13 for freq in counter.values()):
            print(f"삭제된 문장: {text}")  # 삭제된 문장 출력
            continue  # 해당 문장은 저장하지 않음

        filtered_lines.append(line.strip())

# 필터링된 내용 새 파일에 저장
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered_lines))

print(f"총 {len(filtered_lines)}개의 문장 저장 완료 (삭제된 문장 제외).")
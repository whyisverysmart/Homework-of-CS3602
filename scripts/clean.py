import json

def get_data(data):
    utt_id = data["utt_id"]
    manual_transcript = data["manual_transcript"]
    asr_1best = data["asr_1best"]
    semantic = data["semantic"]
    for triple in semantic:
        for char in triple[2]:
            if char not in asr_1best:
                print(asr_1best, triple[2])
                return None
    
    return {
        "utt_id": utt_id,
        "manual_transcript": manual_transcript,
        "asr_1best": asr_1best,
        "semantic": semantic
    }

dataset = json.load(open("data/train.json", 'r', encoding="utf8"))
target_json = []
for di, datas in enumerate(dataset):
    clean_data = []
    for data in datas:
        cleaned_data = get_data(data)
        if cleaned_data is not None:
            clean_data.append(cleaned_data)
    
    if len(clean_data) > 0:
        target_json.append(clean_data)

json.dump(target_json, open("data/train_new.json", 'w', encoding="utf8"), indent=4, ensure_ascii=False)
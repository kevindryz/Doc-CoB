import os, json

if __name__ == '__main__':
    save_path = '/mnt/NAS_SHARE/20240806/6650/datasets/raw_data/vrdu/output/DeepForm-kno_template-train_10-test_5-valid_100-SD_2-test_predictions.json'
    split_path = '/mnt/NAS_SHARE/20240806/6650/datasets/raw_data/vrdu/ad-buy-form/few_shot-splits/DeepForm-kno_template-train_10-test_5-valid_100-SD_2.json'
    output_res = {
        "meta": [],
        "results": {}
    }
    ad_file = '/mnt/NAS_SHARE/20240806/6650/datasets/raw_data/vrdu/calculate_kv/ad-buy-form.json'
    with open(ad_file, 'r') as f:
        ad_data = json.load(f)
    with open(split_path, 'r') as f:
        json_data = json.load(f)
    test_list = json_data['test']
    for pdf_name in test_list:
        output_res['results'][pdf_name] = []
        name = pdf_name.split(".pdf")[0]
        datas = ad_data[name]
        for data in datas:
            key = data['key']
            value = data['value']
            if (not isinstance(key, list)): output_res['results'][pdf_name].append([key, [value, [], []]])
    

    json.dump(output_res, open(save_path, 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
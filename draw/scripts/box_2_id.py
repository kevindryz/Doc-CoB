import json, glob, os
from tqdm import tqdm
import copy
import random
random.seed(42)


def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            # Load each line as a JSON object
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def save_to_jsonl(list_of_dicts, filename):
    # 确保输入是一个列表，并且列表中的每一项都是字典
    if not isinstance(list_of_dicts, list) or not all(
        isinstance(d, dict) for d in list_of_dicts
    ):
        raise ValueError("Input must be a list of dictionaries")

    # 打开文件并逐行写入字典
    with open(filename, "w", encoding="utf-8") as file:
        for dictionary in list_of_dicts:
            json_line = json.dumps(dictionary, ensure_ascii=False)
            file.write(json_line + "\n")

def json_to_dict_by_image_path(json_data):
    result_dict = {}
    for entry in json_data:
        image_path = entry["image"].split("/")[-1][37:]
        if image_path not in result_dict:
            result_dict[image_path] = []
        result_dict[image_path].append(entry)
    
    return result_dict

def normalize_box(bbox, width, height):
    bbox[0] = int((bbox[0] / width) * 1000)
    bbox[1] = int((bbox[1] / height) * 1000)
    bbox[2] = int((bbox[2] / width) * 1000)
    bbox[3] = int((bbox[3] / height) * 1000)
    return bbox

def post_process(result, item, is_test):
    new_data = []
    for line in result:
        if line is not None:
            if not isinstance(line, list):
                line = [line]
            for li in line:
                new_item = copy.deepcopy(item)
                if not is_test:
                    new_item["conversations"] = [
                        {"from": "human", "value": li["prompt"]},
                        {"from": "gpt", "value": li["answer"]},
                    ]
                else:
                    # new_item['image_path']=item['image']
                    new_item["prompt"] = li["prompt"]
                    new_item["gt"] = li["answer"]
                new_item['image_path']=item['image']
                if "draw_box" in li:
                    new_item["draw_box"] = li["draw_box"]
                new_data.append(new_item)
    return new_data
            
template_lis = [
    "What is the box ID corresponding to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Can you provide the box ID corresponding to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Which box ID is associated with <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Please give the box ID for <box>[[{x1}, {y1}, {x2}, {y2}]]</box>.",
    "What box ID corresponds to the coordinates <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Could you tell me the box ID related to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "What is the corresponding box ID for <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Identify the box ID linked to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>.",
    "What's the box ID associated with <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Could you provide the box ID that corresponds to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "What box ID matches the coordinates <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Please specify the box ID that aligns with <box>[[{x1}, {y1}, {x2}, {y2}]]</box>.",
    "What is the unique box ID for <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "What's the corresponding box ID for the box defined by <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Can you identify the box ID linked to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "What is the corresponding box ID of <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Please provide the box ID associated with <box>[[{x1}, {y1}, {x2}, {y2}]]</box>.",
    "What is the box ID that corresponds to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "Could you supply the box ID for <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?",
    "What box ID corresponds to <box>[[{x1}, {y1}, {x2}, {y2}]]</box>?"
]
response_template = """[UNUSED_TOKEN_0]{idx}[UNUSED_TOKEN_1]"""

def process_box_2_id(new_total_boxes, old_to_new_dict, width, height, random_sample = 5):
    random_sample = min(random_sample, len(new_total_boxes))
    new_id = list(old_to_new_dict.values())
    new_id = random.sample(new_id, random_sample)

    result = []
    for idx in new_id:
        sampel_box = normalize_box(new_total_boxes[idx-1], width, height)
        template=random.sample(template_lis,1)[0]
        prompt = template.format(
            x1 = sampel_box[0],
            y1 = sampel_box[1],
            x2 = sampel_box[2],
            y2 = sampel_box[3],
        )
        response = response_template.format(
            idx = idx
        )
        result.append({
            "prompt": prompt,
            "answer": response
        })
    return result



def process_item(ori_item, is_test):
    item = copy.deepcopy(ori_item)
    new_total_boxes = item["new_total_boxes"]
    old_to_new_dict = item["ori_to_new_dict"]
    try:
        width = item["image_size"]["width"]
        height = item["image_size"]["height"]
    except Exception as e:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(item["image"]) as img:
            width, height = img.size

    result = process_box_2_id(new_total_boxes, old_to_new_dict, width, height)

    new_data = post_process(result, ori_item, is_test)

    return new_data


if __name__ == "__main__":

    liss=glob.glob('./241227_gptput/*')
    for pat in liss:
        if '_newid.jsonl' in pat and 'train' in os.path.basename(pat) and '241230_PubLayNet_train_id_background_drawed_10_newid' in os.path.basename(pat):
            print(pat)

            data = read_jsonl_file(pat)

            is_test = False

            image_path_dict = json_to_dict_by_image_path(data)
            
            new_data = []
            for key, item_list in tqdm(image_path_dict.items()):
                item = item_list[0]
                tmp_data = process_item(item, is_test)
                new_data.extend(tmp_data)
            
            save_to_jsonl(new_data, f"./{os.path.basename(pat).split('0')[0]}_box_2_id.jsonl")



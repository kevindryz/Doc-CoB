from box_2_id import read_jsonl_file, save_to_jsonl, post_process, normalize_box
from question_id_2_role import replace_boxes_with_dict
import copy,os
import random,glob
import traceback
from tqdm import tqdm
random.seed(42)

template = """Which red box in the given image contains the answer to the following question '{question}'? Use the box ID near the red box to answer the question.
Below is the correspondence between box ID and box coordinates: {ocr_json}."""

answer_template = """ I believe the most important {box_word} for answering the question {be_word}: {boxes_str}. Therefore, I think the answer to the question is: {answer}."""

box_id_template = "[UNUSED_TOKEN_0]{idx}[UNUSED_TOKEN_1]"

bbox_template = """<box>[[{x1}, {y1}, {x2}, {y2}]]</box>"""

def process_ocr_json_for_prompt(ocr_json):
    new_json = {}
    for index, item in enumerate(ocr_json):
        new_json[index] = {
            "box":item["box"],
            "text":item["text"]
        }
    return new_json

def process_id_2_coord(bbox):
    new_json = {}
    for index, item in enumerate(bbox):
        new_json[box_id_template.format(idx = index + 1)] = bbox_template.format(
            x1 = item[0],
            y1 = item[1],
            x2 = item[2],
            y2 = item[3],
        )
    return new_json

def normalize_box_list(bbox_list, width, height):
    result = []
    for box in bbox_list:
        result.append(normalize_box(box, width, height))
    return result

def parse_gpt_output_dict(gpt_output, old_to_new_dict):
    # 重新组织成字典 并且换下标
    try:
        result = {
            "HELPFUL BOX": [],
            "CONFUSION BOX": []
        }
        if gpt_output == []:
            return result
        
        for key,value in gpt_output.items():
            for item in value:
                reason = item[1]
                reason = replace_boxes_with_dict(reason, old_to_new_dict).replace("<box>","[UNUSED_TOKEN_0]").replace("</box>","[UNUSED_TOKEN_1]")
                idx = old_to_new_dict[str(item[0])]

                result[key].append({
                    "idx": idx,
                    "reason": reason
                })
        return result
    except:
        return {
            "HELPFUL BOX": [],
            "CONFUSION BOX": []
        }

def process_answer(gpt_output, old_to_new_dict, answer, is_single=True):
    try:
        if is_single:
            # box_list = random.sample(gpt_output["HELPFUL BOX"],1)
            box_list = [gpt_output["HELPFUL BOX"][0]]
        else:
            box_list = gpt_output["HELPFUL BOX"] + gpt_output["CONFUSION BOX"]
        box_str_list = []
        for box in box_list:
            box_str_list.append(box_id_template.format(
                idx = box["idx"]
            ))
        return answer_template.format(
            box_word = "box" if len(box_str_list)==1 else "boxes",
            be_word = "is" if len(box_str_list)==1 else "are",
            boxes_str = ", ".join(box_str_list),
            answer = answer
        )
    except Exception as e:
        print(e)
        return None
    

def process_item(ori_item, is_test, is_single):
    item = copy.deepcopy(ori_item)
    try:
        width = item["image_size"]["width"]
        height = item["image_size"]["height"]
    except Exception as e:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(item["image"]) as img:
            width, height = img.size
    
    question = item['conversations'][0]['value']
    if isinstance(item["conversations"][1]["value"], list):
        gt_answer = max(item["conversations"][1]["value"], key=len)
    else:
        gt_answer = item["conversations"][1]["value"]

    boxes = item["new_total_boxes"]
    boxes = normalize_box_list(boxes, width, height)
    box_json = process_id_2_coord(boxes)

    prompt = template.format(
        question = question,
        ocr_json = box_json
    )

    old_to_new_dict = item["ori_to_new_dict"]
    gpt_output = item["gpt_output"]
    gpt_output = parse_gpt_output_dict(item["gpt_output"], old_to_new_dict)

    answer = process_answer(gpt_output, old_to_new_dict, gt_answer, is_single)

    result = [{
        "prompt": prompt,
        "answer": answer
    }]
    
    if not is_test and answer is None:
        return []

    new_data = post_process(result, ori_item, is_test)

    return new_data


if __name__ == "__main__":
    liss=glob.glob('/mnt/workspace/20240806/6650/dieyu/dataset_processor/mineru_ocr/box_data/241227_gptput/*')
    liss=['/mnt/workspace/20240806/6650/dieyu/dataset_processor/mineru_ocr/box_data/241227_gptput/241227_DUDE_val_id_background_drawed_10_newid.jsonl']
    for patt in liss:
        if '_newid.jsonl' in patt:
            if 'test' in os.path.basename(patt):
                is_test = True
            elif 'val' in os.path.basename(patt):
                is_test = True
            elif 'train' in os.path.basename(patt):
                is_test = False
            print(patt,is_test)
            data = read_jsonl_file(patt)
            new_data = []
            for item in tqdm(data):
                tmp_data = process_item(item, is_test)
                new_data.extend(tmp_data)

            save_to_jsonl(new_data, f"/home/moye.my/InternVL/aaa_congcong/241231_dataset/{os.path.basename(patt)}.jsonl")

    


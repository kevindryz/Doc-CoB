from box_2_id import read_jsonl_file, save_to_jsonl, post_process, normalize_box
from question_id_2_role import replace_boxes_with_dict
from one_stage_reasoning import process_ocr_json_for_prompt, process_id_2_coord, normalize_box_list, parse_gpt_output_dict
import copy
import random, glob, os
from tqdm import tqdm
import json
import tempfile
from pathlib import Path
from space_latout import construct_latin_prompt
import traceback
from multiprocessing import Pool, cpu_count
random.seed(42)

template = """{question} The answer is probably found within the red box. The content of the red box is as follows:
{content}"""

# template = """{question}"""
# template = """{question} Please pay attention to the red boxes you previously selected in the earlier round. The answer is probably found within these areas. Answer the question using a single word or phrase."""

template_wo_content = "{question} The answer is most likely within the red box. Answer the question using a single word or phrase." 

box_id_template = "[UNUSED_TOKEN_0]{idx}[UNUSED_TOKEN_1]"

bbox_template = """<box>[[{x1}, {y1}, {x2}, {y2}]]</box>"""

def move_to_top_left(ocr_data):
    # 找到最小的 x 和 y 值
    min_x = float('inf')
    min_y = float('inf')
    
    # 遍历所有的 box
    for entry in ocr_data:
        x, y, _, _ = entry['box']
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

    # 将所有的框移动到左上角
    for entry in ocr_data:
        entry['box'] = [coord - min_x if i % 2 == 0 else coord - min_y for i, coord in enumerate(entry['box'])]
        
        for word in entry['words']:
            word['box'] = [coord - min_x if i % 2 == 0 else coord - min_y for i, coord in enumerate(word['box'])]

def find_completely_covered_text(bbox, ocr_result, coverage_threshold=0.7):
    covered_texts = []

    x_min1, y_min1, x_max1, y_max1 = bbox

    def calculate_area(x_min, y_min, x_max, y_max):
        return max(0, x_max - x_min) * max(0, y_max - y_min)

    def calculate_intersection_area(bbox1, bbox2):
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2

        x_min_overlap = max(x_min1, x_min2)
        y_min_overlap = max(y_min1, y_min2)
        x_max_overlap = min(x_max1, x_max2)
        y_max_overlap = min(y_max1, y_max2)

        return calculate_area(x_min_overlap, y_min_overlap, x_max_overlap, y_max_overlap)

    for item in ocr_result:
        item_bbox = item['box']
        try:
            intersection_area = calculate_intersection_area(bbox, item_bbox)
            item_area = calculate_area(*item_bbox)
            if intersection_area / item_area >= coverage_threshold:
                covered_texts.append(item)
        except:
            pass

    return covered_texts


def process_content(ocr_json, gpt_output, new_total_boxes, is_single, is_test, item):
    try:
        if not is_test:
            if is_single:
                box_list = [gpt_output["HELPFUL BOX"][0]]
            else:
                box_list = gpt_output["HELPFUL BOX"] + gpt_output["CONFUSION BOX"]
                # box_list = sorted(box_list, key=lambda box: box['idx'])
        else:
            box_list = item['s1_result']
        text_list = []
        coord_list = []
        for box in box_list:
            coord = new_total_boxes[box["idx"]-1]
            coord_list.append(coord)
            text_list.extend(find_completely_covered_text(coord,ocr_json))
        move_to_top_left(text_list)

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            json_path = Path(tmp_file.name)
            json.dump(text_list, tmp_file, ensure_ascii=False)

        latin_prompt = construct_latin_prompt(str(json_path), sort_bbox=True, use_ocr_res_type="line", add_newline_between_lines=False, construct_type=1)

        return latin_prompt, coord_list
    except Exception as e:
        print(e)
        # traceback.print_exc()
        return None, None



def process_item(ori_item, is_test, is_single):
    item = copy.deepcopy(ori_item)
    
    question = item['conversations'][0]['value']
    if isinstance(item["conversations"][1]["value"], list):
        gt_answer = max(item["conversations"][1]["value"], key=len)
    else:
        gt_answer = item["conversations"][1]["value"]
    
    old_to_new_dict = item["ori_to_new_dict"]
    gpt_output = item["gpt_output"]
    gpt_output = parse_gpt_output_dict(item["gpt_output"], old_to_new_dict)
    new_total_boxes = item["new_total_boxes"]

    ocr_json = json.load(open(item["ocr_path"].replace('workspace', 'NAS_SHARE')))
    ocr_json = ocr_json["ocr_results"]

    latin_prompt, coord_list = process_content(ocr_json, gpt_output, new_total_boxes, is_single, is_test, item)

    prompt = template.format(
        question = question,
        content = latin_prompt
    )

    result = []
    if latin_prompt is not None:
        result.append({
            "prompt": prompt,
            "answer": gt_answer,
            "draw_box": coord_list
        })
    if not is_test:
        if latin_prompt:
            result.append({
                "prompt": template_wo_content.format(question = question),
                "answer": gt_answer,
                "draw_box": coord_list
            })
        # if coord_list:
        #     result[-1]["draw_box"] = coord_list
    elif len(result) == 0:
        result.append({
            "prompt": question + " Answer the question using a single word or phrase.",
            "answer": gt_answer
        })

    new_data = post_process(result, ori_item, is_test)

    return new_data

def process_item_wrapper(args):
    return process_item(*args)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    # 根目录
    base_pat=os.path.join(data_dir, "s1")
    # 存s1_output的路径
    s1_output_files_lis=f'{base_pat}/s1_output'
    write_pat=f'{base_pat}/Intermediate_results_s1'

    liss=glob.glob(f'{write_pat}/*')

    for pat in liss:
        # if ('DUDE' in pat): continue
        if 'train' in os.path.basename(pat):
            is_test = False
            continue
        elif 'test' in os.path.basename(pat) or 'val' in os.path.basename(pat):
            is_test = True
            # continue
            
        print(pat)
        data = read_jsonl_file(pat)
        is_single = False
        # Prepare arguments for the process_item function
        args = [(item, is_test, is_single) for item in data]

        with Pool(cpu_count()) as pool:
            # Use tqdm to show progress bar
            new_data = []
            for result in tqdm(pool.imap_unordered(process_item_wrapper, args), total=len(data)):
                new_data.extend(result)
            # save_to_jsonl(new_data, f"{base_pat}/Intermediate_results_s1_draw/{os.path.basename(pat)}")
            save_to_jsonl(new_data, f"{base_pat}/todraw/{os.path.basename(pat)}")
        

        new_data = []
        for item in tqdm(data):
            tmp_data = process_item(item, is_test, is_single)
            new_data.extend(tmp_data)

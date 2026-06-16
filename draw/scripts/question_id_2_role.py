from box_2_id import read_jsonl_file, save_to_jsonl, post_process
from tqdm import tqdm
import re, glob, os
import copy,random
import traceback

template_list = [
    "Would you mind elaborating on the purpose and logic behind [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in response to the question, '{question}'",
    "Could you elucidate the function and rationale of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in determining the answer to the question, '{question}'",
    "Can you provide details on the significance and reasoning behind [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in the context of the question, '{question}'",
    "Could you specify the role and the logic of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] when it comes to answering '{question}'",
    "Would you clarify the importance and thought process involving [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] for answering the question, '{question}'",
    "Please explain the reasoning and role of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in resolving the question, '{question}'",
    "Could you please clarify the role and reasoning of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in answering the question, '{question}'",
    "Would you please outline the purpose and reasoning concerning [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in relation to the question, '{question}'",
    "Can you detail what role [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] plays in addressing the question, '{question}'",
    "Please shed light on the function and logic of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] associated with answering the question, '{question}'",
    "Can you explain how [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] is used to answer the question, '{question}'",
    "Might you provide an explanation about how [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] is relevant to finding the answer to the question, '{question}'",
    "Can you delve into the reasoning and purpose of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in the process of answering '{question}'",
    "Would you give a breakdown of the significance and logic behind [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] in the context of answering the question, '{question}'",
    "Could you offer clarification on how [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1]'s role is important in answering the question, '{question}'",
    "Could you describe the role and reasoning associated with [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] when answering the question about '{question}'",
    "Might you articulate the role and thinking behind using [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] for the question, '{question}'",
    "Would you describe the function and importance of [UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] when addressing the question, '{question}'"
]


helpful_template = """[UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] is genuinely helpful to answer the question. The reason is that {reason}"""

confusion_template = """[UNUSED_TOKEN_0]{box_id}[UNUSED_TOKEN_1] is likely to cause confusion in answering the question. The reason is that {reason}"""

def replace_boxes_with_dict(text, mapping):
    def replace_match(match):
        # 找到每个匹配的数字
        num = int(match.group(1))
        # 用字典中的值替换它
        return f"<box>{mapping[str(num)]}</box>"

    # 使用正则表达式替换所有 <box>number</box> 结构
    new_text = re.sub(r"<box>(\d+)</box>", replace_match, text)

    return new_text

def replace_boxes_with_dict_first(text, mapping):
    def replace_match(match):
        # 找到每个匹配的数字
        num = int(match.group(1))
        # 用字典中的值替换它
        return f"<box>{mapping[num]}</box>"

    # 使用正则表达式替换第一个 <box>number</box> 结构
    new_text = re.sub(r"<box>(\d+)</box>", replace_match, text, count=1)

    return new_text

def reverse_dict(original_dict):
    # 使用字典推导式来反转字典
    reversed_dict = {v: k for k, v in original_dict.items()}
    return reversed_dict

def process_q_id_2_role(old_to_new_dict, question, gpt_output):
    new_2_old_dict = reverse_dict(old_to_new_dict)
    result = []
    for key,value in gpt_output.items():
        for item in value:
            # # 先映射成原来de
            # reason = replace_boxes_with_dict_first(item[1], new_2_old_dict)
            
            # #再映射回来，因为他只动了第一个box
            reason = item[1]
            reason = replace_boxes_with_dict(reason, old_to_new_dict).replace("<box>","[UNUSED_TOKEN_0]").replace("</box>","[UNUSED_TOKEN_1]")


            idx = old_to_new_dict[str(item[0])]
            template=random.sample(template_list,1)[0]
           
            new_dict = {
                "prompt": template.format(
                    box_id=idx,
                    question=question
                )
            }
            if key == "HELPFUL BOX":
                new_dict["answer"] = helpful_template.format(
                    box_id=idx,
                    reason=reason
                )
            elif key == "CONFUSION BOX":
                new_dict["answer"] = confusion_template.format(
                    box_id=idx,
                    reason=reason
                )
            result.append(new_dict)
    return result


def process_item(ori_item, is_test):
    item = copy.deepcopy(ori_item)
    # print(item["image_path"])
    old_to_new_dict = item["ori_to_new_dict"]
    # question = item["prompt"].split("\n")[-1].replace("Analyze and think progressively and answer the question using a single word or phrase: ","").strip()
    question=item['conversations'][0]['value']

    gpt_output = item["gpt_output"]
    if isinstance(gpt_output,list):
        pass
    else:

        result = process_q_id_2_role(old_to_new_dict, question, gpt_output)

        new_data = post_process(result, ori_item, is_test)

        return new_data


if __name__ == "__main__":
    liss=glob.glob('/mnt/workspace/20240806/6650/dieyu/dataset_processor/mineru_ocr/box_data/241227_gptput/*')
    for pat in liss:
        if '_newid.jsonl' in pat and 'train' in os.path.basename(pat):
            print(pat)
            data = read_jsonl_file(pat)
    
            is_test = False

            new_data = []
            for item in tqdm(data):
                try:
                    tmp_data = process_item(item, is_test)
                    new_data.extend(tmp_data)
                except:
                    pass



        
            save_to_jsonl(new_data, f"/home/moye.my/InternVL/aaa_congcong/241229_dataset/{os.path.basename(pat).split('0')[0]}_q_id_2_role.jsonl")
            print(f"/home/moye.my/InternVL/aaa_congcong/241229_dataset/{os.path.basename(pat).split('0')[0]}_q_id_2_role.jsonl")

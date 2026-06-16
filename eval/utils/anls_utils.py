import json
import numpy as np
import os
import copy
# from icecream import ic
from typing import Optional
from textvqa_eval import TextVQAAccuracyEvaluator
import textdistance

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
meta_dir_pre = os.path.join(data_dir, "meta")

def get_origin_id(origin_test_file):
    id_dict = {}
    with open(origin_test_file, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            image_name = os.path.basename(json_data['image_path']).split('_page')[0]
            if image_name not in id_dict: id_dict[image_name] = []
            id_dict[image_name].append([json_data['conversations'][0]['value'], json_data['questionId']])
    return id_dict

def exclude_qa(mode="test"):
    exclude_qa_list = []
    if mode=="test":
        curdataset_path = os.path.join(data_dir, "test_v1.0.json")
    with open(curdataset_path, 'r', encoding='utf-8') as file:
        curdataset = json.load(file)['data']
    for item in curdataset:
        if item['questionId'] in [679, 58467, 58715, 58780, 59870, 60015, 61084, 62529, 62530, 62532, 5434, 5462, 5541, 6093, 56626, 56627, 39010, 53693, 65412, 1639, 59950, 60919, 61198, 62419]:
            exclude_qa_list.append({
                "image": os.path.basename(item['image']),
                "question": item['question']
            })
    # return []
    return exclude_qa_list

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def normANLS(s1,s2):
    s1 = str(s1)
    s2= str(s2)
    dist = levenshtein_distance(s1.lower().strip(),s2.lower().strip())
    length = max(len(s1),len(s2))
    value =  0.0 if length == 0 else float(dist) / float(length) 
    return value 

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        json_data = json.loads(line)
        data.append(json_data)
    return data

def get_dude_mapping():
    path = os.path.join(data_dir, "DUDE_val.json")
    with open(path, 'r') as f:
        json_data = json.load(f)
    new_dict = {}
    for json_item in json_data:
        docid = json_item['docId']
        answers = json_item['answers']
        key = docid
        for answer in answers:
            key += answer
        new_dict[key] = json_item
    return new_dict

def read_jsonl_byId(path, id_list):
    data = []
    num_dict = {}
    id_list = sorted(id_list)
    with open(path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      for idx, line in enumerate(lines):
        json_data = json.loads(line)
        num_dict[idx] = json_data
    for idx in id_list:
        data.append(num_dict[idx])
    return data

def check_path(curpath):
    if os.path.exists(curpath): return curpath
    curpath = curpath.replace("/mnt/NAS_SHARE/","/mnt/mynas/")
    if os.path.exists(curpath): return curpath
    curpath = curpath.replace("/mnt/mynas/","/home/admin/workspace/aop_lab/")
    if os.path.exists(curpath): return curpath
    curpath = curpath.replace("/home/admin/workspace/aop_lab/","/mnt/workspace/")
    if os.path.exists(curpath): return curpath
    print(f"file not exit {curpath}")   
    raise OSError(f"File not found: {curpath}")

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            data.append(json_data)
    return data

def get_pdf_list(data):
    files = []
    name_dict = {}
    for item in data:
        name = f"{os.path.basename(item['image_path']).split('_page')[0]}.pdf"
        files.append(name)
        if (name not in name_dict): name_dict[name] = []
        q = item['conversations'][0]['value']
        entity = q.split('for the ')[1]
        name_dict[name].append(entity)
    unique_files = list(set(files))
    return unique_files, name_dict

def filter_original_data(original_data, train_kv_dict, test_kv_dict):
    output_res = []
    for item in original_data:
        tem = {
            'filename': item['filename'],
            'file_path': item['file_path'],
            'ocr': item['ocr'],
            'annotations': item['annotations'],
        }
        # train_kv_dict 中key是有.pdf的
        if (item['filename'] in train_kv_dict): v_list = train_kv_dict[item['filename']]
        elif (item['filename'] in test_kv_dict): v_list = test_kv_dict[item['filename']]
        else: continue

        for anno in item['annotations']:
            key = anno[0]
            if (key not in v_list):
                tem['annotations'].remove(anno)
        output_res.append(tem)
    return output_res

def format_jsonl(jsonl_output, test_kv_dict):
    output_res = {
        "meta": [],
        "results": {}
    }
    for item in jsonl_output:
        if 'image' in item: image = os.path.basename(item['image']).split('_page')[0]
        else: image = os.path.basename(item['image_path']).split('_page')[0]
        # 如果没有肯定是因为uuid
        if (f'{image}.pdf' not in test_kv_dict):
            if (len(image) > 9): image2 = image[8:]
            else: image2 = image
            if (f'{image2}.pdf' not in test_kv_dict):
                image = image[37:]
                if (f'{image}.pdf' not in test_kv_dict):
                    image = image[37:]
            else: image = image[8:]
        name = f'{image}.pdf'
        v_list = test_kv_dict[name]
        # 看v_list中的值哪个在q中
        q = item['conversations'][0]['value']
        res = item.get('result', None) 
        if res is None: res = 'none'
        for v in v_list:
            if (f'for the {v}' not in q): continue
            if name not in output_res['results']: output_res['results'][name] = []
            new_list = [
                v,
                [
                    res,
                    [],
                    []
                ]
            ]
            output_res['results'][name].append(new_list)
            break
    return output_res

def read_jsonl_byName(path, name_dict):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      for idx, line in enumerate(lines):
        json_data = json.loads(line)
        if (json_data['name'].replace('.pdf', '') not in name_dict): continue
        answer_list = name_dict[json_data['name'].replace('.pdf', '')]
        res = {"name": json_data['name'].replace('.pdf', ''), "language": "en", "annotations": [], "split": "test"}
        for answer_item in json_data['annotations']:
            if (answer_item['key'] in answer_list): res['annotations'].append(answer_item)
        data.append(res)
    return data

def save_jsonl(data, path):
    with open(path,'w')as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) +'\n')
    # print('save %d samples(imgs) to %s ' % (len(data), path))

def fill_pred_item(questionId, answers):
    new_dict = {
        "questionId": questionId,
        "answers": answers if isinstance(answers, list) else [answers],
        "answers_confidence": [1]
    }
    return new_dict

def evaluateANLS(ans_list, anls_threshold = 0.5):
    # anls_list = []
    # for predict_pair in ans_list:
    #     val = predict_pair["answer"]
    #     possible_vals = predict_pair["annotation"]
    #     print(val, possible_vals)
    #     best_score = max([textdistance.levenshtein.normalized_similarity(val.lower().strip(), pos.lower().strip())
    #                     for pos in possible_vals])
    #     if 1 - anls_threshold >= best_score:
    #         best_score = 0.0            
    #     anls_list.append(best_score)
    # return np.float64(sum(anls_list) / len(anls_list))
    # anls_threshold = 0.5
    anls_list = []
    true_list, false_list = [], []
    for predict_pair in ans_list:
        if isinstance(predict_pair["answer"], list):
            # assert len(predict_pair["answer"]) == 1
            answer = predict_pair["answer"][0].strip()
        elif isinstance(predict_pair["answer"], str):
            answer = predict_pair["answer"].strip()
            
        value_list = []
        if isinstance(predict_pair["annotation"], list):
            gt_list = predict_pair["annotation"]
            for gt_single in gt_list:
                # if gt_single.strip().lower() in answer.strip().lower():
                #     value_list.append(0)
                value_list.append(normANLS(gt_single,answer))
        elif isinstance(predict_pair["annotation"], str):
            gt_single = predict_pair["annotation"]
            value_list.append(normANLS(gt_single,answer))
            
        question_result = 1 - min(value_list)

        if (question_result < anls_threshold) :
            question_result = 0
            tem = predict_pair['pre_res']
            # tem.pop('total_boxes', None)
            # tem.pop('new_total_boxes', None)
            # tem.pop('ori_to_new_dict', None)
            # tem.pop('image', None)
            # tem.pop('image_path', None)
            false_list.append(tem)
        else: true_list.append(predict_pair['pre_res'])
        anls_list.append(question_result)
    return np.mean(anls_list), true_list, false_list

def are_numbers_identical(str1, str2):
    # 将字符串拆分为数字，并将其转换为集合
    set1 = set(str1.split(','))
    set2 = set(str2.split(','))

    return set1 == set2

def evaluate_exact_match_accuracy(entries):
    scores = []
    true_list, false_list = [], []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        if isinstance(elem['answer'], list):
            elem['answer'] = elem['answer'][0]

        # old_score = max([
        #     (1.0 if
        #      (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
        #     for ann in elem['annotation']
        # ])
        for ann in elem['annotation']:
            if '[UNUSED_TOKEN_0]' in ann:
                print("UNUSED_TOKEN_0 error")
        if '[UNUSED_TOKEN_0]' in elem['answer']: print("UNUSED_TOKEN_0 error")
        score = max([
            (1.0 if
             (are_numbers_identical(elem['answer'].strip().lower(), ann.strip().lower())) else 0.0)
            for ann in elem['annotation']
        ])
        # if(score != old_score):
        #     print("ok")
        if (score > 0.5):
            true_list.append(elem['pre_res'])
        else: false_list.append(elem['pre_res'])
        scores.append(score)
    return sum(scores) / len(scores), true_list, false_list

def processed_jsonl(jsonl_file, flag):
    if (flag == 0):
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
        result = []
        wrong_result = []
        for line in lines:
            data = json.loads(line)
            data_dict = {}
            # data_dict['image_path'] = data['image_path']
            data_dict['question'] = data['conversations'][0]['value']
            if type(data['result']) == list: data_dict['answer'] = data['result'][0]
            else: data_dict['answer'] = data['result']
            # if (data_dict['answer'][-1] == '.'): data_dict['answer'] = data_dict['answer'][:-1]
            data_dict['annotation'] = data['conversations'][1]['value']
            # data_dict['question_id'] = data['questionId']
            result.append(data_dict)
            if (data_dict['annotation'] != data_dict['answer']): wrong_result.append(data_dict)
        return result, wrong_result
    else:
        lines = jsonl_file
        result = []
        wrong_result = []
        for data in lines:
            data_dict = {}
            # data_dict['image_path'] = data['image_path']
            data_dict['question'] = data['conversations'][0]['value']
            if type(data['result']) == list: data_dict['answer'] = data['result'][0]
            else: data_dict['answer'] = data['result']
            # if (data_dict['answer'][-1] == '.'): data_dict['answer'] = data_dict['answer'][:-1]
            data_dict['annotation'] = data['conversations'][1]['value']
            # data_dict['question_id'] = data['questionId']
            result.append(data_dict)
            if (data_dict['annotation'] != data_dict['answer']): wrong_result.append(data_dict)
        return result, wrong_result

def eval_due(dataset_name, pred_path, gt_path, id_list, name_dict):
    metrics = ['F1']
    preds = read_jsonl(pred_path)
    # preds = read_jsonl_byId(pred_path, id_list)
    # print(name_dict)
    gts = read_jsonl_byName(gt_path, name_dict)
    # gts = read_jsonl(gt_path)
    # print('pred %d, gt %d' % (len(preds), len(gts)))
    from due_evaluator.due_evaluator import DueEvaluator
    for metric in metrics:
        evaluator = DueEvaluator(reference=gts,
                                answers=preds,
                                ignore_case=True,
                                metric=metric)
        general_scorer, label_scorers = evaluator._evalute()
        return general_scorer.score()
        # ic('Overall %s:%.4f' % (metric, general_scorer.score()))
        """for label, scorer in label_scorers.items():
             print('%s %s:%.4f' % (label, metric, scorer.score()))"""

def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def count_numbers_in_string(s):
    # 使用逗号或全角逗号分割
    numbers = s.replace('，', ',').split(',')
    # 清除空字符串并统计有效数字
    numbers = [num.strip() for num in numbers if num.strip().isdigit()]
    return len(numbers)

class evaluateScore:
    def __init__(self):
        pass

    def evalANLS(self, jsonl_file_path, anls_threshold=0.5):
        all_result = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                all_result.append(json.loads(line))
        outputs = []
        exclude_qa_list = exclude_qa()
        exclude_count=0
        invalid_num = 0
        for onepair in all_result:
            if 'image_path' in onepair:
                onepair['image'] = onepair['image_path']
            flag = False
            for one_ex in exclude_qa_list:
                if one_ex['image'].strip().lower() in onepair['image'].strip().lower() and one_ex['question'].strip().lower() in onepair['conversations'][0]['value'].strip().lower():
                    exclude_count += 1
                    flag = True
            if flag: continue
            annotation = onepair['conversations'][1]['value']
            if annotation == "":
                invalid_num += 1
                continue
            if isinstance(annotation, str) and annotation[0]=='[':
                annotation = eval(annotation)
            
            outputs.append({
                # 'question': onepair['conversations'][0]['value'],
                'answer': onepair['result'],
                'annotation': annotation,
                'pre_res': onepair
            })
        anls_res, true_list, false_list = evaluateANLS(outputs, anls_threshold)
        return anls_res
    
    def evalVQAScore(self, jsonl_file_path, annotation_path):
        evaluator = TextVQAAccuracyEvaluator()
        annotation = json.load(open(annotation_path, 'r'))['annotations']
        all_result = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                all_result.append(json.loads(line))
        question_id2answers = {}
        for item in annotation:
            question_id = item['question_id']
            answers = [answer['answer'] for answer in item['answers']]
            question_id2answers[question_id] = answers
        for item in all_result:
            item['pred_answer'] = item['result']
            item['gt_answers'] = question_id2answers[item['question_id']]
        accuracy = evaluator.eval_pred_list(all_result)

        return accuracy

    def evalACC(self, jsonl_file_path, multi_area = False):
        all_result = []
        none_answer = 0
        has_answer = 0
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                all_result.append(json.loads(line))
        merged_outputs = []
        replace_unused_token = 0
        for item in all_result:
            # count = count_numbers_in_string(item['conversations'][1]['value'])
            if isinstance(item['result'], list):
                for i in range(len(item['result'])):
                    item['result'][i] = item['result'][i].replace('[UNUSED_TOKEN_0]','').replace('[UNUSED_TOKEN_1]','')
            else:
                item['result'] = item['result'].replace('[UNUSED_TOKEN_0]','').replace('[UNUSED_TOKEN_1]','')
            if isinstance(item['conversations'][1]['value'], list):
                assert(len(item['result'])) == 1
                for i in range(len(item['conversations'][1]['value'])):
                    item['conversations'][1]['value'][i] = item['conversations'][1]['value'][i].replace('[UNUSED_TOKEN_0]','').replace('[UNUSED_TOKEN_1]','')
            else:
                item['conversations'][1]['value'] = item['conversations'][1]['value'].replace('[UNUSED_TOKEN_0]','').replace('[UNUSED_TOKEN_1]','')

            if item['conversations'][1]['value'] == "":
                none_answer += 1
                continue
            has_answer += 1
            merged_outputs.append({"annotation": item['conversations'][1]['value'],"answer": item['result'], 'pre_res': item})
        accuracy, true_list, false_list = evaluate_exact_match_accuracy(merged_outputs)
        
        return accuracy, none_answer, has_answer

    def evalRelaxedAccuracy(self, jsonl_file_path):
        merged_outputs, wrong_merged_outputs = processed_jsonl(jsonl_file_path, 0)
        scores = []
        for elem in merged_outputs:
            if isinstance(elem['annotation'], str):
                elem['annotation'] = [elem['annotation']]
            score = max([
                relaxed_correctness(elem['answer'].strip(), ann)
                for ann in elem['annotation']
            ])
            scores.append(score)
        return sum(scores) / len(scores)

    def eval_F1(self, llm_pred_path = None, meta_dir = './meta', dataset_name = 'DeepForm', split = 'test'):
        """
        reformat results by LLM for due-benchmark evaluation 

        """
        # print(llm_pred_path)
        meta_dir = meta_dir_pre
        assert dataset_name in ['DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 'DeepForm', 'KleisterCharity', 'TabFact']
        # ic(dataset_name)
        if dataset_name == 'DeepForm':
            dataset_categories = ['advertiser', 'flight_from', 'flight_to', 'gross_amount', 'contract_num']
        elif dataset_name == 'KleisterCharity':
            dataset_categories = ['address__post_town',
                            'address__postcode',
                            'address__street_line',
                            'charity_name',
                            'charity_number',
                            'income_annually_in_british_pounds',
                            'report_date',
                            'spending_annually_in_british_pounds']
        
        preds = []
        id_list = []
        origin_test_file = os.path.join(meta_dir_pre, 'DeepForm_test.jsonl')
        origin_id_dict = get_origin_id(origin_test_file)
        with open(llm_pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                assert len(json_data['conversations']) == 2
                question = json_data['conversations'][0]['value']
                if 'image' in json_data: 
                    name = os.path.basename(json_data['image']).split('_page')[0]
                elif 'image_path' in json_data: 
                    name = os.path.basename(json_data['image_path']).split('_page')[0]
                if (len(name) > 9): 
                    name2 = name[8:]
                else: 
                    name2 = name
                if (name2 not in origin_id_dict):
                    if (name not in origin_id_dict): 
                        name = name[37:]
                        if (name not in origin_id_dict): name = name[37:]
                else: 
                    name = name2
                if ("id" in json_data):
                    question_list = origin_id_dict[name]
                    for item in question_list:
                        if (item[0] in question):
                            json_data['id'] = item[1]
                            break
                    preds.append({
                                'id':json_data['id'],
                                'question': question,
                                'answer':str(json_data['result']).strip().replace('\n', '')})
                    id_list.append(json_data['id'])
                elif ('questionId' in json_data):
                    question_list = origin_id_dict[name]
                    for item in question_list:
                        if (item[0] in question):
                            json_data['questionId'] = item[1]
                            break
                    preds.append({
                                'questionId': json_data['questionId'],
                                'question': question,
                                'answer':str(json_data['result']).strip().replace('\n', '')})
                    id_list.append(json_data['questionId'])
                else:
                    preds.append({
                                # 'name':json_data['image'][0],
                                'question': question,
                                'answer':str(json_data['result']).strip().replace('\n', '')})
        if ("id" in json_data): 
            preds = sorted(preds, key=lambda x: x['id'])
        if ("questionId" in json_data): 
            preds = sorted(preds, key=lambda x: x['questionId'])
        meta_path = os.path.join(meta_dir, dataset_name, split, 'metadata.jsonl')
        meta_data = read_jsonl_byId(meta_path, id_list)
        # meta_data = read_jsonl(meta_path)

        name_dict = {}
        for meta_item in meta_data:
            name = os.path.splitext(os.path.basename(meta_item['file_name']))[0]
            if (name not in name_dict): 
                name_dict[name] = []
            question = json.loads(meta_item['ground_truth'])['gt_parses'][0]['question']
            parts = question.split()
            last_part = parts[-1][:-1]
            name_dict[name].append(last_part)

        # print(meta_data)
        
        # ic(len(meta_data), len(preds))
        assert len(meta_data) == len(preds)
        for i in range(len(meta_data)):
            preds[i]['name'] = meta_data[i]['file_name'].split('/')[-1].split('.pdf')[0]
            # for ie task, covert category question to the category
            if dataset_name in ['DeepForm', 'KleisterCharity']:
                cate_question = json.loads(meta_data[i]['ground_truth'])['gt_parses'][0]['question']
                for cate in dataset_categories:
                    if cate in cate_question:
                        preds[i]['question'] = cate
                        break
            # for qa task, copy question is necessary, question in preds can have some minor revisions
            # keep quesiton consistent with gt file is necessary for due eveluation
            else:
                preds[i]['question'] = json.loads(meta_data[i]['ground_truth'])['gt_parses'][0]['question']

        # reorganize preds to 1 line means QA pairs or category-value pairs of 1 image
        due_preds = []
        img = {}
        for i in range(len(preds)):
            pred = preds[i]
            if 'name' not in img: # start img
                img['name'] = pred['name']
                img['annotations'] = []
            elif pred['name'] != img['name']: # save previous img results and init a new one
                due_preds.append(copy.deepcopy(img))
                img = {}
                img['name'] = pred['name']
                img['annotations'] = []

            # for ie task, if the answer is none, drop the category-value pair
            if dataset_name not in ['DeepForm', 'KleisterCharity'] or pred['answer'] != 'None':
                img['annotations'].append({'key':pred['question'], 'values':[{'value':pred['answer']}]})
            
            if i == len(preds)-1:
                due_preds.append(copy.deepcopy(img))
        if dataset_name == 'TabFact':
            due_preds = add_tabfact_missing_img(due_preds, meta_dir)
        save_path = llm_pred_path.replace('.jsonl', '_due.jsonl')
        save_jsonl(due_preds, save_path)

        gt_path = os.path.join(meta_dir, dataset_name, split, 'document.jsonl')
        score = eval_due(dataset_name, save_path, gt_path, id_list, name_dict)
        os.remove(save_path)
        return score

    def eval_ANLS_dude(self, jsonl_file_path, anls_threshold=0.5):
        # 用来处理gt
        gt_name, val_name = 'DUDE_val_gt_tem.json', 'DUDE_val.json'
        gt_dir = os.path.join(data_dir, 'DUDEeval')
        gt_dict = get_dude_mapping()
        res_pred = []
        res_gt = {
            "dataset_name": "DUDE Dataset",
            "dataset_version": "0.1",
            "data": []
        }
        # 首先要处理一下jsonl文件，同时存一下json文件，这样就可以直接用原本评估代码，之后再删就行
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    json_obj = json.loads(line)
                    image_path = json_obj['image'] if 'image' in json_obj else json_obj['image_path']
                    docid = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
                    if 'ocr_path' in json_obj: docid = os.path.splitext(os.path.basename(json_obj['ocr_path']))[0].split('_')[0]
                    # 问题因为会变，还是用答案比较合适，如果是list的话，每一个都取出来字符串拼接
                    answers = json_obj['conversations'][1]['value']
                    key = docid
                    if type(json_obj['prompt']) == list: json_obj['prompt'] = json_obj['prompt'][0]
                    if 'image' in json_obj: uuid_name = os.path.basename(json_obj['image'])
                    else: uuid_name = os.path.basename(json_obj['image_path'])
                    if (len(uuid_name) > 36 and uuid_name[8] == '-' and uuid_name[13] == '-' and uuid_name[18] == '-' and uuid_name[23] == '-'):
                        if (json_obj['prompt'].startswith('Please read the image, paying attention to the red boxes labeled with numbers')): 
                            key = os.path.splitext(os.path.basename(json_obj['image'])[37:])[0].split('_')[0]
                    else: 
                        if (json_obj['prompt'].startswith('Please read the image, paying attention to the red boxes labeled with numbers')): 
                            key = os.path.splitext(os.path.basename(json_obj['image'])[9:])[0].split('_')[0]
                        # try:
                        #     if (json_obj['prompt'].startswith('Please read the image, paying attention to the red boxes labeled with numbers')): 
                        #         key = os.path.splitext(os.path.basename(json_obj['image'])[9:])[0].split('_')[0]
                        # except:
                        #     print()
                        #     exit()
                    # print(key)
                    if isinstance(answers, list):
                        for answer in answers: key += answer
                    else: key += answers
                    questionId = gt_dict[key]['questionId']
                    result = json_obj['result']
                    pred_item = fill_pred_item(questionId, result)
                    res_pred.append(pred_item)
                    res_gt['data'].append(gt_dict[key])
                except json.JSONDecodeError as e:
                    print('ERROR')
        pred_path = os.path.join(gt_dir, 'submissions', val_name)
        gt_name_path = os.path.join(gt_dir, 'gt', gt_name)
        json.dump(res_pred, open(pred_path, 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(res_gt, open(gt_name_path, 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
        from evaluate_submission import solve
        return solve(gt_name_path, pred_path, anls_threshold)

        # 1. 处理输出的jsonl,处理成相应json的形式，保存在output文件夹中
    def eval_VRDU_F1(self, jsonl_file_path, dataset_name):
        # 清空输出文件夹，方便生成处理后的输出文件
        processed_output_path = os.path.join(data_dir, 'vrdu', 'output')
        delete_files_in_folder(processed_output_path)

        # 存一下split的文件，同时解决命名问题 split_name 作为处理后保存的名字.    output中的文件加上-test_predictions
        train_jsonl = os.path.join(meta_dir_pre, f'VRDU_{dataset_name}-form_train.jsonl')
        test_jsonl = os.path.join(meta_dir_pre, f'VRDU_{dataset_name}-form_test.jsonl')
        train_data, test_data = read_jsonl(train_jsonl), read_jsonl(test_jsonl)
        train_pdf_list, train_kv_dict = get_pdf_list(train_data)
        test_pdf_list, test_kv_dict = get_pdf_list(test_data)
        split_msg = {
            "train": train_pdf_list,
            "valid": [],
            "test": test_pdf_list
        }
        split_name = f'DeepForm-mixed_template-train_{len(train_pdf_list)}-test_{len(test_pdf_list)}-valid_0-SD_0.json'
        few_shot_path = os.path.join(data_dir, f'vrdu/{dataset_name}-form/few_shot-splits')
        if (dataset_name != 'ad-buy'): split_name.replace('DeepForm', 'FARA-lv2')
        save_path = os.path.join(few_shot_path, split_name)
        json.dump(split_msg, open(save_path, 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
        # 重组一份dataset.jsonl，只包含train_kv_dict和test_kv_dict中有的信息
        pre_datasets_file = os.path.join(data_dir, f'vrdu/{dataset_name}-form/main/dataset_original.jsonl')
        original_data = read_jsonl(pre_datasets_file)
        processed_original = filter_original_data(original_data, train_kv_dict, test_kv_dict)
        save_path = os.path.join(data_dir, f'vrdu/{dataset_name}-form/main/dataset.jsonl')
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in processed_original:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 输出的文件
        jsonl_output = read_jsonl(jsonl_file_path)
        output_res = format_jsonl(jsonl_output, test_kv_dict)
        save_path = os.path.join(processed_output_path, split_name.replace('SD_0.json', 'SD_0-test_predictions.json'))
        json.dump(output_res, open(save_path, 'w', encoding="utf-8"), ensure_ascii=False, indent=2)

        from vrdu.evaluate import vrdu_solve
        base_dirpath = os.path.join(data_dir, f'vrdu/{dataset_name}-form')
        extraction_path = os.path.join(data_dir, 'vrdu', 'output')
        eval_output_path = os.path.join(data_dir, 'vrdu', 'output.csv')
        vrdu_solve(dataset_name, base_dirpath, extraction_path, eval_output_path)
        import csv
        micro_f1_value, macro_f1_value = None, None
        with open(eval_output_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                micro_f1_value = row['metric-micro_f1']
                macro_f1_value = row['metric-macro_f1']
        return micro_f1_value, macro_f1_value   
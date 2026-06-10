import sys, json, os, glob
sys.path.append('./utils')
from anls_utils import evaluateScore

evalclass = evaluateScore()

def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

if __name__ == '__main__':
    pat_lis = glob.glob('./*')
    for item in pat_lis:
        jsonl_file_path = item
        if ('FUNSD' in jsonl_file_path or 'SROIE' in jsonl_file_path or 'DocVQA' in jsonl_file_path or 'Info' in jsonl_file_path):
            res = evalclass.evalANLS(jsonl_file_path=jsonl_file_path, anls_threshold=0.5)
            print(os.path.basename(jsonl_file_path), res, 'ANLS')
        elif ('VRDU' in jsonl_file_path):
            divide = 'registration'
            if ('ad-buy' in jsonl_file_path): divide = 'ad-buy'
            res1, res2 = evalclass.eval_VRDU_F1(jsonl_file_path = jsonl_file_path, dataset_name = divide)
            print(os.path.basename(jsonl_file_path), res1, res2, 'VRDU_F1')
        elif ('Deep' in jsonl_file_path):
            res = evalclass.eval_F1(llm_pred_path = jsonl_file_path, meta_dir = '/mnt/NAS_SHARE/20240806/6650/dieyu/myutils/meta', dataset_name = "DeepForm", split = 'test')
            print(os.path.basename(jsonl_file_path), res, 'F1')
        elif ('DUDE' in jsonl_file_path):
            res = evalclass.eval_ANLS_dude(jsonl_file_path=jsonl_file_path, anls_threshold=0.5)
            print(os.path.basename(jsonl_file_path), res, 'ANLS_DUDE')
        elif ('Chart' in jsonl_file_path):
            res = evalclass.evalACC(jsonl_file_path=jsonl_file_path)
            print(os.path.basename(jsonl_file_path), res)

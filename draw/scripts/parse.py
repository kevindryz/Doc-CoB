import json,glob,os,re
from box_2_id import read_jsonl_file, save_to_jsonl

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")

# 根目录
base_pat=os.path.join(data_dir, "s1")
# 存s1_output的路径
s1_output_files_lis=f'{base_pat}/s1_output'
write_pat=f'{base_pat}/Intermediate_results_s1'


pat_lis=glob.glob(f'{data_dir}/gptput_ori/*')

for filepath in pat_lis:
    if 'test' in filepath or 'val' in filepath:
        print(filepath)
        d={}
        his = {}
        s1_img = {}
        if 'VRDU' not in filepath:
            parts = os.path.basename(filepath).split('_')[1]
            if 'DUDE' in filepath:
                output_name=f'241227_{parts}_val_id_background_drawed_10_newid.jsonl.jsonl.res.jsonl'
            else:
                output_name=f'241227_{parts}_test_id_background_drawed_10_newid.jsonl.jsonl.res.jsonl'
        else:
            parts = os.path.basename(filepath).split('_')
            desired_part = '_'.join(parts[1:3])
            output_name=f'241227_{desired_part}_test_id_background_drawed_10_newid.jsonl.jsonl.res.jsonl'
        data_output=read_jsonl_file(f'{s1_output_files_lis}/{output_name}')
        print(f'{s1_output_files_lis}/{output_name}')
        # exit()
        for entry in data_output:
            pattern = r"swer to the following question '(.*?)'\? Use the box ID near the red box to"
            match = re.search(pattern, entry['prompt'])
            question = match.group(1)
            # output_id = f'{os.path.basename(entry["image"])}_{question}'
            output_id = f'{os.path.basename(entry["image"])[37:]}_{question}'
            if isinstance(entry['result'], list):
                d[output_id] = entry['result'][0]
            else: d[output_id]= entry['result']
            # his[output_id] = entry['history']
            # s1_img[output_id] = entry['image']
        # print(len(d))
        process_data=[]
        data=read_jsonl_file(filepath)
        for entry in data:
            output_id=f'{os.path.basename(entry["image"])}_{entry["conversations"][0]["value"]}'
            
            pattern = r'\[UNUSED_TOKEN_0\](\d+)\[UNUSED_TOKEN_1\]'
            input_string=d[output_id]
            new_list = [{"idx":int(x)} for x in re.findall(pattern, input_string)]
            entry['s1_result'] = new_list
            # entry['history'] = his[output_id]
            # entry['s1_img'] = s1_img[output_id]
            process_data.append(entry)
        write_filename=f'{write_pat}/{os.path.basename(filepath)}'
        print(write_filename)
        save_to_jsonl(process_data,write_filename)
        # exit()
            

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-27 14:22:26
# @Author  : Zhaoqing Zhu (zhuzhaoqing.zzq@alibaba-inc.com)
# @Link    : https://code.alibaba-inc.com/GPT-X/DocGenerator
# @Version : 改造latin-prompt, refine版本为“construct_type”=2的情况；改造功能：1. 避免各文字块之间粘连；2. 距离较远的lines之间添加空行"\n"; 3. 考虑same line的新判断条件：如果ocr line里面有"line_idx“这一字段，则根据此区分same line; 4. 区分english doc还是chinese doc，chinese doc的每个char对应2个空格
import os, sys, json
import numpy as np
import re
from functools import cmp_to_key

class OctSortHelper:

    # 计算水平方向相邻类型
    @staticmethod
    def calc_x_type(a, b):
        x_type = 0
        minx_a, maxx_a = a[0], a[0]+a[2]
        minx_b, maxx_b = b[0], b[0]+b[2]

        start_left = 0
        if minx_a < minx_b:
            start_left = 1
        elif minx_a > minx_b:
            start_left = -1
        end_right = 0
        if maxx_a > maxx_b:
            end_right = 1
        elif maxx_a < maxx_b:
            end_right = -1

        if maxx_a<minx_b+1e-4 and maxx_a<maxx_b-1e-4:
            x_type = 1 #left
        elif minx_a>maxx_b-1e-4 and minx_a>minx_b+1e-4:
            x_type = 2 #right
        elif start_left==1 and end_right==-1:
            x_type = 3 #near left
        elif start_left==-1 and end_right==1:
            x_type = 4 #near right
        elif start_left>=0 and end_right>=0:
            x_type = 5 #contain
        elif start_left<=0 and end_right<=0:
            x_type = 6 #inside
        else:
            x_type = 0

        return x_type

    # 计算垂直方向相邻类型
    @staticmethod
    def calc_y_type(a, b):
        y_type = 0
        miny_a, maxy_a = a[1], a[1]+a[3]
        miny_b, maxy_b = b[1], b[1]+b[3]

        start_up = 0
        if miny_a < miny_b:
            start_up = 1
        elif miny_a > miny_b:
            start_up = -1
        end_down = 0
        if maxy_a > maxy_b:
            end_down = 1
        elif maxy_a < maxy_b:
            end_down = -1
        
        if maxy_a<miny_b+1e-4 and maxy_a<maxy_b-1e-4:
            y_type = 1 #up
        elif miny_a>maxy_b-1e-4 and miny_a>miny_b+1e-4:
            y_type = 2 #down
        elif start_up==1 and end_down==-1:
            y_type = 3 #near up
        elif start_up==-1 and end_down==1:
            y_type = 4 #near down
        elif start_up>=0 and end_down>=0:
            y_type = 5 #contain
        elif start_up<=0 and end_down<=0:
            y_type = 6 #inside
        else:
            y_type = 0

        return y_type

    # 按照从上到下从左到右比较阅读顺序
    @staticmethod
    def cmp_reading_order_udlr(a, b, thres=0.5):
        minx_a, miny_a, maxx_a, maxy_a = a[0], a[1], a[0]+a[2], a[1]+a[3]
        minx_b, miny_b, maxx_b, maxy_b = b[0], b[1], b[0]+b[2], b[1]+b[3]

        x_type, y_type = OctSortHelper.calc_x_type(a, b), OctSortHelper.calc_y_type(a, b)

        y_near_rate = 0.0
        if y_type==3:
            y_near_rate = (maxy_a-miny_b) / min(maxy_a-miny_a, maxy_b-miny_b)
        elif y_type==4:
            y_near_rate = (maxy_b-miny_a) / min(maxy_a-miny_a, maxy_b-miny_b)

        if y_type==1:
            return 1
        elif y_type==2:
            return -1
        elif y_type==3:
            if x_type in [2, 4]:
                if y_near_rate < thres:
                    return 1
                else:
                    return -1
            else:
                return 1
        elif y_type==4:
            if x_type in [1, 3]:
                if y_near_rate < thres:
                    return -1
                else:
                    return 1
            else:
                return -1
        else:
            if x_type==1 or x_type==3:
                return 1
            elif x_type==2 or x_type==4:
                return -1
            else:
                center_y_diff = abs(0.5*(miny_a+maxy_a) - 0.5*(miny_b+maxy_b))
                max_h = max(maxy_a-miny_a, maxy_b-miny_b)
                if center_y_diff/max_h < 0.1:
                    if (minx_a+maxx_a) < (minx_b+maxx_b):
                        return 1
                    elif (minx_a+maxx_a) > (minx_b+maxx_b):
                        return -1
                    else:
                        return 0
                else:
                    if (miny_a+maxy_a) < (miny_b+maxy_b):
                        return 1
                    elif (miny_a+maxy_a) > (miny_b+maxy_b):
                        return -1
                    else:
                        return 0

    # 按照从右到左从上到下比较阅读顺序
    @staticmethod
    def cmp_reading_order_rlud(a, b):
        minx_a, miny_a, maxx_a, maxy_a = a[0], a[1], a[0]+a[2], a[1]+a[3]
        minx_b, miny_b, maxx_b, maxy_b = b[0], b[1], b[0]+b[2], b[1]+b[3]

        x_type, y_type = OctSortHelper.calc_x_type(a, b), OctSortHelper.calc_y_type(a, b)

        x_near_rate = 0.0
        if x_type==3:
            x_near_rate = (maxx_a-minx_b) / min(maxx_a-minx_a, maxx_b-minx_b)
        elif x_type==4:
            x_near_rate = (maxx_b-minx_a) / min(maxx_a-minx_a, maxx_b-minx_b)

        if x_type==1:
            return -1
        elif x_type==2:
            return 1
        elif x_type==3:
            if x_near_rate<0.4:
                return -1
            else:
                if y_type==1 or y_type==3:
                    return 1
                elif y_type==2 or y_type==4:
                    return -1
                else:
                    if (a.miny+a.maxy)<(b.miny+b.maxy):
                        return 1
                    elif (a.miny+a.maxy)>(b.miny+b.maxy):
                        return -1
                    else:
                        return 0
        elif x_type==4:
            if x_near_rate<0.4:
                return 1
            else:
                if y_type==1 or y_type==3:
                    return 1
                elif y_type==2 or y_type==4:
                    return -1
                else:
                    if (miny_a+maxy_a)<(miny_b+maxy_b):
                        return 1
                    elif (miny_a+maxy_a)>(miny_b+maxy_b):
                        return -1
                    else:
                        return 0
        else:
            if y_type==1 or y_type==3:
                return 1
            elif y_type==2 or y_type==4:
                return -1
            else:
                if (miny_a+maxy_a)<(miny_b+maxy_b):
                    return 1
                elif (miny_a+maxy_a)>(miny_b+maxy_b):
                    return -1
                else:
                    return 0
    
    # blocks : a list of [x,y,w,h]
    @staticmethod
    def sort_ocr_rects(ocr_rects):
        ocr_rects.sort(key=cmp_to_key(OctSortHelper.cmp_reading_order_udlr))

    @staticmethod
    def sort_standard_ocr_data_cmp(x,y):
        a = [x['box'][0], x['box'][1],x['box'][2]-x['box'][0],x['box'][3]-x['box'][1]]
        b = [y['box'][0], y['box'][1],y['box'][2]-y['box'][0],y['box'][3]-y['box'][1]]
        return -OctSortHelper.cmp_reading_order_udlr(a,b)

    @staticmethod
    def sort_standard_ocr(ocr_data):
        return sorted(ocr_data, key=cmp_to_key(OctSortHelper.sort_standard_ocr_data_cmp))


class DocSpaceLayout:
    def __init__(self, use_advanced_space_layout=False):
        if use_advanced_space_layout:
            self.use_py_space = False
        else:
            self.use_py_space = True

        self.use_advanced_space_layout = use_advanced_space_layout

    
    @staticmethod
    def box4point_to_box2point(box4point):
        # bounding box = [x0, y0, x1, y1, x2, y2, x3, y3]
        all_x = [box4point[2 * i] for i in range(4)]
        all_y = [box4point[2 * i + 1] for i in range(4)]
        box2point = [min(all_x), min(all_y), max(all_x), max(all_y)]
        return box2point
    
    @staticmethod
    def is_same_line(box1, box2):
        """
        Params:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        
        box1_midy = (box1[1] + box1[3]) / 2
        box2_midy = (box2[1] + box2[3]) / 2

        if (box1_midy < box2[3] and box1_midy > box2[1] and
            box2_midy < box1[3] and box2_midy > box1[1]):
            return True
        else:
            return False

    @staticmethod
    def union_box_list(bbox_list):
        boxes = np.array(bbox_list)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # 计算合并后的box坐标
        union_box = [np.min(x1), np.min(y1), np.max(x2), np.max(y2)]
        return union_box
    
    @staticmethod
    def union_box(box1, box2):
        """
        Params:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])

        return [x1, y1, x2, y2]

    @staticmethod
    def boxes_sort(boxes):
        """
        Params:
            boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
        sorted_id = sorted(
            range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))

        return sorted_id

    @staticmethod
    def boxes_sort_from_left_to_right(boxes):
        # 使用enumerate获取每个元素的原始索引，并与元素一起排序
        sorted_box_list_with_idx = sorted(enumerate(boxes), key=lambda x: x[1][0])

        # 从排序后的列表中提取排序后的元素和它们的原始索引
        sorted_box_list = [item[1] for item in sorted_box_list_with_idx]
        sorted_indices = [item[0] for item in sorted_box_list_with_idx]

        return sorted_box_list, sorted_indices


    def space_layout(self, texts, boxes):
        """
        Params:
            texts: ocr 文本行string [text1, text2, ...]
            boxes: ocr 文本行坐标 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """

        line_boxes = []
        line_texts = []
        max_line_char_num = 0
        line_width = 0
        # print(f"len_boxes: {len(boxes)}")
        while len(boxes) > 0:
            line_box = [boxes.pop(0)]
            line_text = [texts.pop(0)]
            char_num = len(line_text[-1])
            line_union_box = line_box[-1]
            while len(boxes) > 0 and self.is_same_line(line_box[-1], boxes[0]):
                line_box.append(boxes.pop(0))
                line_text.append(texts.pop(0))
                char_num += len(line_text[-1])
                line_union_box = self.union_box(line_union_box, line_box[-1])
            line_boxes.append(line_box)
            line_texts.append(line_text)
            if char_num >= max_line_char_num:
                max_line_char_num = char_num
                line_width = line_union_box[2] - line_union_box[0]
        
        # print(line_width)

        char_width = line_width / max_line_char_num
        # print(char_width)
        if char_width == 0:
            char_width = 1

        space_line_texts = []
        for i, line_box in enumerate(line_boxes):
            space_line_text = ""
            for j, box in enumerate(line_box):
                left_char_num = int(box[0] / char_width)
                space_line_text += " " * (left_char_num - len(space_line_text))
                space_line_text += line_texts[i][j]
            space_line_texts.append(space_line_text)


        doc_str = "\n".join(space_line_texts)
        return doc_str, space_line_texts

    def expand_string_from_middle(self, input_str, target_length, space_increment):
        # 首先确保输入的字符串长度不超过目标长度
        if len(input_str) >= target_length:
            return input_str  # 如果已经达到或超过目标长度，则直接返回原字符串

        # 找到中间点的位置
        mid_index = len(input_str) // 2

        # 从中间往左右两边扩展空格
        left_index, right_index = mid_index, mid_index
        if len(input_str) % 2 == 0:  # 如果是偶数长度的字符串，左右索引会有所不同
            left_index -= 1

        # 从中间向左右两边扩展，直到达到目标长度
        while len(input_str) < target_length:
            # 向左边扩展空格
            if left_index >= 0 and input_str[left_index] == ' ':
                input_str = input_str[:left_index] + ' ' * space_increment + input_str[left_index:]
                left_index -= space_increment
                right_index += space_increment

            # 向右边扩展空格
            if right_index < len(input_str) and input_str[right_index] == ' ' and len(input_str) < target_length:
                input_str = input_str[:right_index] + ' ' * space_increment + input_str[right_index:]
                right_index += space_increment

            # 更新左右索引
            left_index -= 1
            right_index += 1

            # 如果左右索引超出范围，则停止循环
            if left_index < 0 and right_index >= len(input_str):
                break

        # # 如果扩展后的字符串长度超过了目标长度，进行适当的截断
        # if len(input_str) > target_length:
        #     input_str = input_str[:target_length]

        return input_str

    def compute_mix_ch_eng_string_len(self, mix_string):
        # compute the len of string that mix chinese and english, 1 chinese char = 2 lens, 1 other char = 1 len
        total_length = 0
        for char in mix_string:
            # 检查当前字符是否为中文字符
            if '一' <= char <= '鿿':
                total_length += 2
            else:
                total_length += 1
        return total_length

    def space_layout_refine(self, texts, boxes, text_line_idx, add_newline_between_lines=False):
        """
        Params:
            texts: ocr 文本行string [text1, text2, ...]
            boxes: ocr 文本行坐标 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            text_line_idx: 文本行的idx[idx1, idx2, ...], 若无该字段则是[]
        """

        line_boxes = []
        line_texts = []
        max_line_char_num = 0
        line_width = 0
        # min_char_width = 1001
        line_max_word_num, line_max_word_width = 0, 0
        # print(f"len_boxes: {len(boxes)}")
        while len(boxes) > 0:
            max_word_num = 0
            min_char_width = 1001
            line_box = [boxes.pop(0)]
            line_text = [texts.pop(0)]
            char_num = self.compute_mix_ch_eng_string_len(line_text[-1])

            line_width = line_box[-1][2] - line_box[-1][0]
            char_width = line_width/char_num
            if char_width < min_char_width:
                min_char_width = char_width
                max_word_num = char_num
                max_word_width = line_width
                # print(f"line_text: {line_text[-1]}, max_word_num: {max_word_num}, max_word_width: {max_word_width}")

            line_union_box = line_box[-1]
            box_idx = len(text_line_idx) - len(boxes) - 1 
            # import ipdb; ipdb.set_trace()
            while len(boxes) > 0 and ((text_line_idx != [] and text_line_idx[box_idx] == text_line_idx[box_idx+1]) or (text_line_idx == [] and self.is_same_line(line_box[-1], boxes[0]))):
                line_box.append(boxes.pop(0))
                line_text.append(texts.pop(0))
                char_num = self.compute_mix_ch_eng_string_len(line_text[-1])
                line_width = line_box[-1][2] - line_box[-1][0]

                char_width = line_width/char_num
                if char_width < min_char_width:
                    min_char_width = char_width
                    max_word_num = char_num
                    max_word_width = line_width
                    # print(f"line_text: {line_text[-1]}, max_word_num: {max_word_num}, max_word_width: {max_word_width}")
                box_idx = len(text_line_idx) - len(boxes) - 1 
                line_union_box = self.union_box(line_union_box, line_box[-1])

            # 归为同一行的，再从左至右sort
            line_box, sorted_indicess = self.boxes_sort_from_left_to_right(line_box)
            line_text = [line_text[sort_idx] for sort_idx in sorted_indicess]

            line_word_num = sum([self.compute_mix_ch_eng_string_len(word_text) for word_text in line_text])

            if line_word_num >= max_line_char_num:
                max_line_char_num = line_word_num
                line_max_word_num = max_word_num
                line_max_word_width = max_word_width
                # print(f"line_text: {line_text}, line_word_num: {line_word_num}, line_max_word_num: {line_max_word_num}, line_max_word_width: {line_max_word_width}")

            line_boxes.append(line_box)
            line_texts.append(line_text)
            
        
        # print(line_width)
        # import ipdb; ipdb.set_trace()

        # char_width = line_width / max_line_char_num
        # char_width = min_char_width
        char_width = line_max_word_width / line_max_word_num
        # print("char_width: ", char_width)
        if char_width == 0:
            char_width = 1

        space_line_texts = []
        pre_union_box = []
        # for i, line_box in enumerate(line_boxes):
        #     space_line_text = ""
        #     for j, box in enumerate(line_box):
        #         # mid_box = int(box[0] + (box[2] - box[0]) / 2)
        #         # mid_char_num = int(len(line_texts[i][j]) / 2)
        #         # left_char_num = int(mid_box / char_width - mid_char_num)
        #         target_text_length = int((box[2] - box[0]) //char_width)
        #         if len(line_texts[i][j]) < target_text_length:
        #             space_nums = line_texts[i][j].count(' ')
        #             if space_nums > 0:
        #                 space_increment = int((target_text_length - (len(line_texts[i][j]) - space_nums)) / space_nums + 1)
        #                 line_texts[i][j] = self.expand_string_from_middle(line_texts[i][j], target_text_length, space_increment)
        #         # if space_nums == 0:
        #         mid_box = int(box[0] + (box[2] - box[0]) / 2)
        #         mid_char_num = int(len(line_texts[i][j]) / 2)
        #         left_char_num = int(mid_box / char_width - mid_char_num)
        #         # left_char_num = int(box[0] / char_width)

        #         space_line_text += " " * max(1, (left_char_num - len(space_line_text)))
        #         space_line_text += line_texts[i][j]
        #     space_line_texts.append(space_line_text)
        for i, line_box in enumerate(line_boxes):
            space_line_text = ""
            
            if add_newline_between_lines:
                # import ipdb; ipdb.set_trace()
                cur_union_box = self.union_box_list(line_box)
                if pre_union_box != []:
                    pre_box_height = pre_union_box[3] - pre_union_box[1]
                    cur_box_height = cur_union_box[3] - cur_union_box[1]
                    height_diff = cur_union_box[1] - pre_union_box[3]
                    if height_diff > min(pre_box_height, cur_box_height):
                        space_line_text += "\n"
                pre_union_box = cur_union_box
            # import ipdb; ipdb.set_trace()
            for j, box in enumerate(line_box):
                left_char_num = round(box[0] / char_width)
                space_line_text += " " * max(1, (left_char_num - self.compute_mix_ch_eng_string_len(space_line_text.strip("\n"))))
                space_line_text += line_texts[i][j]
            

            space_line_texts.append(space_line_text)


        doc_str = "\n".join(space_line_texts)
        return doc_str, space_line_texts


def construct_latin_prompt(ocr_input, sort_bbox=True, construct_type=1, use_ocr_res_type="line", add_newline_between_lines=False):
    '''
        Parameters:
        sort bbox: 在构造layout prompt之前sort bbox
        use_ocr_res_type: "line": 用文字块进行construct; "word": 用word进行construct
        add_newline_between_lines: 是否在分隔较开的lines之间加空行
        construct_type: 1: norm latin-prompt; 2: refine latin-prompt
    '''
    if ".json" in ocr_input:
        with open(ocr_input) as f:
            ocr_lines_result = json.load(f)
    else:
        ocr_lines_result = ocr_input


    texts = []
    text_boxes = []
    text_line_idx = []

    if sort_bbox:
        ocr_lines_result = OctSortHelper.sort_standard_ocr(ocr_lines_result)

    if len(ocr_lines_result) > 1 and "line_idx" in ocr_lines_result[0]:
        ocr_lines_result.sort(key=lambda item: int(item["line_idx"]))
    # import ipdb; ipdb.set_trace()

    if use_ocr_res_type == "line":
        for line in ocr_lines_result:
            if "line_idx" in line: 
                line_idx = line["line_idx"]
            else:
                line_idx = None
            texts.append(line["text"])
            text_boxes.append(line["box"])
            if line_idx is not None:
                text_line_idx.append(line_idx)
        if text_line_idx != [] and len(text_line_idx) != len(text_boxes):
            exit("Error! Don't align!")
    elif use_ocr_res_type == "word":
        for line in ocr_lines_result:
            if "line_idx" in line: 
                line_idx = line["line_idx"]
            else:
                line_idx = None
            for word in line["words"]:
                texts.append(word["text"])
                text_boxes.append(word["box"])
                if line_idx is not None:
                    text_line_idx.append(line_idx)
        if text_line_idx != [] and len(text_line_idx) != len(text_boxes):
            exit("Error! Don't align!")
    else:
        exit("Error ocr res type \"{}\"!".format(use_ocr_res_type))

    # latin_prompt
    doc_space_layout = DocSpaceLayout()
    if construct_type == 1:
        doc_str, space_line_texts = doc_space_layout.space_layout(texts, text_boxes)
    elif construct_type == 2:
        doc_str, space_line_texts = doc_space_layout.space_layout_refine(texts, text_boxes, text_line_idx, add_newline_between_lines=add_newline_between_lines)

    latin_prompt = "\n".join(space_line_texts)

    return latin_prompt

def construct_plain_text(ocr_input, sort_bbox=True, use_ocr_res_type="line"):
    if ".json" in ocr_input:
        with open(ocr_input) as f:
            ocr_lines_result = json.load(f)
    else:
        ocr_lines_result = ocr_input


    texts = []
    text_boxes = []
    text_line_idx = []

    if sort_bbox:
        ocr_lines_result = OctSortHelper.sort_standard_ocr(ocr_lines_result)

    if len(ocr_lines_result) > 1 and "line_idx" in ocr_lines_result[0]:
        ocr_lines_result.sort(key=lambda item: int(item["line_idx"]))
    # import ipdb; ipdb.set_trace()

    if use_ocr_res_type == "line":
        for line in ocr_lines_result:
            if "line_idx" in line: 
                line_idx = line["line_idx"]
            else:
                line_idx = None
            texts.append(line["text"])
            text_boxes.append(line["box"])
            if line_idx is not None:
                text_line_idx.append(line_idx)
        if text_line_idx != [] and len(text_line_idx) != len(text_boxes):
            exit("Error! Don't align!")
    elif use_ocr_res_type == "word":
        for line in ocr_lines_result:
            if "line_idx" in line: 
                line_idx = line["line_idx"]
            else:
                line_idx = None
            for word in line["words"]:
                texts.append(word["text"])
                text_boxes.append(word["box"])
                if line_idx is not None:
                    text_line_idx.append(line_idx)
        if text_line_idx != [] and len(text_line_idx) != len(text_boxes):
            exit("Error! Don't align!")
    else:
        exit("Error ocr res type \"{}\"!".format(use_ocr_res_type))
    
    return " ".join(texts)

class LatinHelper:
    @staticmethod
    def ocr2lation_prompt(ocr_data):
        if ocr_data == []:
            return ""
        doc_space_layout = DocSpaceLayout()

        texts = list(map(lambda x:x["text"],ocr_data))
        text_boxes = list(map(lambda x:x["box"],ocr_data))
        doc_str, space_line_texts = doc_space_layout.space_layout(texts, text_boxes)

        return doc_str

if __name__ == "__main__":
    # doc_space_layout = DocSpaceLayout()
    # filepath = ("/mnt/common/chuwei.lcw/data/QABank/DocVQA/"
    #     "ori_dataset/test/ocr_results/ysbw0217_14.json")

    # import json
    # with open(filepath, "r") as f:
    #     ocr_lines_result = json.load(f)["recognitionResults"][0]["lines"]

    # words = []
    # word_boxes = []
    # texts = []
    # text_boxes = []

    # for line in ocr_lines_result:
    #     texts.append(line["text"])
    #     text_boxes.append(DocSpaceLayout.box4point_to_box2point(line["boundingBox"]))
    #     for word in line["words"]:
    #         words.append(word["text"])
    #         word_boxes.append(DocSpaceLayout.box4point_to_box2point(word["boundingBox"]))

    # doc_str, space_line_texts = doc_space_layout.space_layout(texts, text_boxes)
    # print(doc_str)

    ocr_path = "/mnt/nlp_ocr_v100/shenyufan/VIE/data/output/ocr_result_set/DOCVQA/test/documents-jycf0227_2.png.json"

    img_path = "/mnt/nlp_ocr_v100/shenyufan/VIE/data/vqa/docvqa/test/images/hrfw0227_24.png"
    ocr_path = "/mnt/nlp_ocr_v100/shenyufan/VIE/data/output/ocr_result_set/DOCVQA/test/documents-hrfw0227_24.png.json"
    latin_prompt = construct_latin_prompt(ocr_path, sort_bbox=True, use_ocr_res_type="word", add_newline_between_lines="True", construct_type=2)
    print("#"*100)
    print(latin_prompt)
    
    '''
    Parameters:
    sort bbox: 在构造layout prompt之前sort bbox
    use_ocr_res_type: "line": 用文字块进行construct; "word": 用word进行construct
    add_newline_between_lines: 是否在分隔较开的lines之间加空行
    construct_type: 1: norm latin-prompt; 2: refine latin-prompt
    '''
    # ocr_path = "/mnt/nlp_ocr_v100/zhuzq/project2/DocGenerator/test/hrfw0227_24.json"
    latin_prompt = construct_latin_prompt(ocr_path, sort_bbox=True, use_ocr_res_type="line", add_newline_between_lines="True", construct_type=2)
    print("#"*100)
    print(latin_prompt)

    ocr_path = "/mnt/nlp_ocr_v100/zhuzq/project2/DocGenerator/test/PPT测试图片.json"
    ocr_path = "/mnt/nlp_ocr_v100/shenyufan/VIE/data/output/archive/0612_docvqa_listanswer/truncated_ocr/test/truncated_69_jycf0227_2..json"
    ocr_path = "/mnt/nlp_ocr_v100/shenyufan/VIE/data/output/archive/0612_docvqa_listanswer/truncated_ocr/test/truncated_49_kqbf0227_1..json"
    ocr_path = "/mnt/nlp_ocr_v100/shenyufan/VIE/data/output/archive/0612_docvqa_listanswer/truncated_ocr/test/truncated_87_fxbw0217_6..json"

    latin_prompt = construct_latin_prompt(ocr_path, sort_bbox=True, use_ocr_res_type="line", add_newline_between_lines="True", construct_type=1)
    print("#"*100)
    print(latin_prompt)

    ocr_path = "/mnt/workspace/workgroup/zhuzq/project2/DocGenerator/test/vehicle_certification_2302__data_pool_0883af7415e4ffa2bc9cbafa0c5f672d.json"
    # ocr_path = "/mnt/workspace/workgroup/zhuzq/project2/DocGenerator/test/vehicle_certification_2302__data_pool_8bf7f5a5c8d472298e245aaa3f21dd6c.json"

    latin_prompt = construct_latin_prompt(ocr_path, sort_bbox=True, use_ocr_res_type="line", add_newline_between_lines="True", construct_type=2)
    # with open(ocr_path) as f:
    #     ocr_results = json.load(f)
    # latin_prompt = construct_latin_prompt(ocr_path)
    print("#"*100)
    print("latin_prompt: \"\n{}\n\"".format(str(latin_prompt)))
    
# Doc-CoB
### 数据集下载
```bash
# 本地路径自行指定
modelscope download --dataset mercury01/Doc-CoB --local_dir ./
```
| 资源 | 链接 |
| ---- | ---- |
| 🤖 | [https://www.modelscope.cn/datasets/mercury01/Doc-CoB](https://www.modelscope.cn/datasets/mercury01/Doc-CoB) |

### 评测数据集部署
1. 下载VRDU评估数据集相关文件；
2. 找到解压后`eval`目录下两个VRDU数据集对应的`.jsonl`文件；
3. 在项目根目录依次创建路径：`data/vrdu/[数据集名称]/main/`；
4. 将下载的`.jsonl`文件分别移入各自对应数据集的`main`文件夹内。
5. 运行`eval/evaluate.py`脚本进行测试, 其中`pat_lis`指定为包含测试结果文件的路径。

示例目录结构参考：
```
eval/
└── data/
    └── vrdu/
        ├── ad-buy-form/
        │   └── main/
        │       └── dataset_original.jsonl
        └── registration-form/
            └── main/
                └── dataset_original.jsonl
```

### 训练脚本使用说明
项目根目录下 `draw` 文件夹包含 `scripts`、`shell` 两个子目录：
1. `scripts`：存放训练所需Python辅助脚本；
2. `shell`：提供两个训练启动脚本
   - `train.sh`：常规训练脚本
   - `train_cot.sh`：CoT范式训练脚本
   - `merge.sh`：模型权重合并脚本

运行脚本前需要根据本地实际环境修改文件内路径参数：
- model：预训练模型存储路径
- dataset：数据集存放目录
- output_dir：训练权重、日志输出目录
- adapters：需要合并的权重路径

其余超参数可根据实验需求自行调整。

目录结构如下
```
draw/
├── scripts/
└── shell/
    ├── train.sh
    ├── train_cot.sh
    └── merge.sh
```

### 画框脚本使用说明

完成数据格式整理后，需按固定顺序运行三段脚本，最终生成S2推理文件与标注边框可视化图片：
1. **parse.py**
    读取S1阶段推理输出数据，统一清洗、转换为模型推理标准输入格式。
    前置要求：S1推理结果处理完成后存入`s1_output`文件夹，参考样例`s1_output/case.jsonl`规范，推理内容统一写入JSONL内`result`字段。(评测s2最终结果时也是如此)
2. **two_stage_reasoning_s2.py**
    基于处理好的S1输出开展第二阶段推理，生成S2结构化推理结果文件。
3. **draw_box.py**
    读取S2推理文件中的坐标标注信息，在原图上绘制对应边框，输出带标注框的可视化图像。

必须依次执行：`parse.py` → `two_stage_reasoning_s2.py` → `draw_box.py`，顺序不可调换，否则会出现数据缺失、绘图失败问题。

目录结构如下
```
draw/
├── scripts/
│   ├── parse.py
│   ├── two_stage_reasoning_s2.py
│   └── draw_box.py
└── shell/
    ├── train.sh
    └── train_cot.sh
```
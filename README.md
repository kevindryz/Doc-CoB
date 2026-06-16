# Doc-CoB
### 数据集部署
1. 下载VRDU评估数据集相关文件；
2. 找到解压后`eval`目录下两个VRDU数据集对应的`.jsonl`文件；
3. 在项目根目录依次创建路径：`data/vrdu/[数据集名称]/main/`；
4. 将下载的`.jsonl`文件分别移入各自对应数据集的`main`文件夹内。

示例目录结构参考：
```
project_root/
└── data/
    └── vrdu/
        ├── ad-buy-form/
        │   └── main/
        │       └── xxx.jsonl
        └── registration-form/
            └── main/
                └── yyy.jsonl
```
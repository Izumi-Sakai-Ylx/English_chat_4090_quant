## 0. 以下均为离线量化

## 1. AWQ 量化
我们采用 transformers 库来进行量化，使用的 version 是 " GEMM "，最终将其量化成 int4 的模型

运行脚本
```
quant-AWQ.py
```

## 2. GPTQ 量化
使用 auto_gptq 库来进行量化，使用 " WikiText "数据集进行校准，同时量化需要指定是 int4 还是 int8

运行脚本
```
quant-GPTQ.py
```

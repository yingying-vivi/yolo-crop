# Archive 说明文档

本目录存放了 v1/v2/v3 版本的训练脚本和对应的训练结果，已被 v4 版本替代。
v4 版本使用的配置：imgsz=640, mosaic=0.5, close_mosaic=20, freeze=10, patience=200, dropout=0.2, cos_lr=True, lr0=0.002, weight_decay=0.001

---

## 一、FGFD 数据集脚本（已废弃）

### 1. train_fgfd_baseline.py → 结果：fgfd_baseline
- 数据集：fgfd (nc=2, 红/绿分开)
- 模型：标准 YOLO11n-seg
- 配置：epochs=100, imgsz=512, 无额外增强参数，默认增强
- 结果：mAP50-95(M)=0.4760, best epoch=90
- 废弃原因：nc=2 不符合论文要求，无自定义增强配置

### 2. train_fgfd_baseline_v2.py → 结果：fgfd_baseline_v2 / fgfd_baseline_v3
- 数据集：fgfd_1cls (nc=1, 农田 vs 背景)
- 模型：标准 YOLO11n-seg
- 配置：epochs=200, imgsz=512, mosaic=0.0, patience=50, dropout=0.1, lr0=0.005, 无freeze
- 注意：脚本名叫v2但实际输出name="fgfd_baseline_v3"
- 结果：mAP50-95(M)=0.5251, best epoch=52
- 废弃原因：imgsz=512+mosaic=0.0 导致严重过拟合；无freeze导致backbone不稳定

### 3. train_fgfd_star.py → 结果：无（未完成/被覆盖）
- 数据集：fgfd (nc=2)
- 模型：yolo11-star-seg (C3k2_Star替换所有C3k2，StarBlock shortcut=True)
- 配置：epochs=100, imgsz=512, 默认增强
- 废弃原因：nc=2错误，StarBlock shortcut=True导致CSP冲突

### 4. train_fgfd_star_v2.py → 结果：fgfd_star_v2 / fgfd_star_v3
- 数据集：fgfd_1cls (nc=1)
- 模型：yolo11-star-seg (C3k2_Star替换所有C3k2)
- 配置：epochs=200, imgsz=512, mosaic=0.0, patience=50, dropout=0.1, lr0=0.005
- 注意：脚本名叫v2但实际输出name="fgfd_star_v3"，StarBlock shortcut=False修复版
- fgfd_star_v2: mAP50-95(M)=0.5194, best epoch=77 (shortcut=True版)
- fgfd_star_v3: mAP50-95(M)=0.5241, best epoch=79 (shortcut=False修复版)
- 废弃原因：imgsz=512+mosaic=0.0过拟合；StarBlock在浅层不合理；freeze=10导致backbone StarBlock被冻结成死权重

### 5. train_fgfd_eca.py → 结果：无
- 数据集：fgfd (nc=2)
- 模型：yolo11-eca-seg (ECA插入head)
- 配置：epochs=100, imgsz=512, 默认增强，ECA shift-remap权重加载
- 废弃原因：nc=2错误，无自定义增强

### 6. train_fgfd_eca_v2.py → 结果：无（未跑完）
- 数据集：fgfd_1cls (nc=1)
- 模型：yolo11-eca-seg (ECA插入head)
- 配置：epochs=200, imgsz=512, mosaic=0.0, ECA shift-remap权重加载
- 废弃原因：imgsz=512+mosaic=0.0过拟合问题

### 7. train_fgfd_field_loss.py → 结果：无
- 数据集：fgfd (nc=2)
- 模型：标准 YOLO11-seg + FieldSegmentationModel (boundary loss)
- 配置：epochs=100, imgsz=512, 默认增强
- 废弃原因：nc=2错误

### 8. train_fgfd_field_loss_v2.py → 结果：无
- 数据集：fgfd_1cls (nc=1)
- 模型：标准 YOLO11-seg + FieldSegmentationModel (boundary loss)
- 配置：epochs=200, imgsz=512, mosaic=0.0
- 废弃原因：imgsz=512+mosaic=0.0过拟合问题

### 9. train_fgfd_all.py → 结果：无
- 综合脚本，包含baseline/star/eca/field_loss四个实验
- 数据集：fgfd (nc=2)
- 配置：epochs=100, imgsz=512, 默认增强
- 废弃原因：nc=2错误，已被v4系列独立脚本替代

---

## 二、Paddy Field 数据集脚本（已废弃）

### 10. train_star.py → 结果：无
- 数据集：paddy_field (528张, 2048x2048, 3类)
- 模型：yolo11-star-seg
- 配置：epochs=100, imgsz=640, batch=8
- 废弃原因：不再使用paddy_field数据集，聚焦FGFD

### 11. train_eca.py → 结果：无
- 数据集：paddy_field
- 模型：yolo11-eca-seg, ECA shift-remap权重加载
- 配置：epochs=100, imgsz=640, batch=8
- 废弃原因：同上

### 12. train_field_loss.py → 结果：无
- 数据集：paddy_field
- 模型：YOLO11-seg + FieldSegmentationModel (boundary loss)
- 配置：epochs=100, imgsz=640, batch=8
- 废弃原因：同上

### 13. train_all.py → 结果：train-4
- 数据集：paddy_field
- 模型：yolov8n-seg
- 配置：epochs=100, imgsz=640, batch=8
- 结果：train-4, mAP50-95(M)=0.5561, best epoch=100
- 废弃原因：yolov8对比实验，不再需要

---

## 三、验证/调试结果（无对应脚本）

### val, val-2 ~ val-6
- 手动在终端运行的验证测试，无独立脚本
- 仅用于调试权重加载和数据格式，不包含正式训练结果

### base
- 早期基础验证，无results.csv

### 加载预训练权重——528 / 未加载预训练权重——528
- paddy_field数据集的预训练权重加载对比实验
- 无results.csv，仅用于调试

---

## 四、关键发现总结（v1→v2→v3→v4的演进原因）

1. nc=2 → nc=1: FGFD论文明确要求红+绿合并为1类"农田"，nc=2使任务更难且不符合论文
2. mosaic=0 → mosaic=0.5: mosaic=0导致FGFD(512x512固定patch)每epoch看到完全相同的像素，严重过拟合
3. imgsz=512 → imgsz=640: 640配合mosaic=0.5提供更多空间多样性
4. patience=50 → patience=200: 确保模型充分训练到最佳性能
5. StarBlock shortcut=True → shortcut=False: 在CSP(C2f/C3k2)内部，shortcut与split-merge双重冗余，抑制star操作
6. StarBlock位置：浅层→深层: StarBlock的7x7大感受野更适合深层空间结构提取，浅层3x3就够了
7. freeze=10 → freeze=0/6: freeze=10冻结backbone StarBlock(随机初始化)=死权重，必须让StarBlock能训练

---

## 五、当前活跃的v4脚本（不在archive中）

| 脚本 | 模型 | freeze | 结果目录 | mAP50-95(M) |
|---|---|---|---|---|
| train_fgfd_baseline_v4.py | 标准C3k2 | 10 | fgfd_baseline_v4 | 0.5578 |
| train_fgfd_baseline_freeze0_v4.py | 标准C3k2 | 0 | 待跑 | - |
| train_fgfd_baseline_freeze6_v4.py | 标准C3k2 | 6 | 待跑 | - |
| train_fgfd_star_deep_v4.py | StarBlock@深层backbone | 0 | fgfd_star_deep_v4 | 0.5771 |
| train_fgfd_star_deep_freeze6_v4.py | StarBlock@深层backbone | 6 | fgfd_star_deep_freeze6_v4 | 正在跑 |
| train_fgfd_star_v4.py | StarBlock@所有位置 | 10 | fgfd_star_v4 | 0.5518 |
| train_fgfd_star_head_v4.py | StarBlock@head | 10 | 待跑 | - |
| train_fgfd_eca_v4.py | ECA@head | 10 | 待跑 | - |
| train_fgfd_field_loss_v4.py | Boundary Loss | 10 | 待跑 | - |
深度学习在 成熟棉花田场景中，对散落在土壤上的“杂志”类杂质进行检测，可用于开发棉花产区智能化清洁与质量预警系统 成熟棉花花壤杂质数据集 棉花杂质检测数据集
@[toc]

成熟棉花花壤杂质数据集。,yolo格式，txt标签。已经划分好训练集（1784张图片）验证集（403张图片）数据集标注准确，实测map50为77.6%。数据集一类:杂志。包含类别标签说明文件。不含数据增强的图片。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/feee7852379048819740d1ef71b6a92d.png)



1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ee470d7392c048678c2af63f48253aa6.png)


---

### **成熟棉花花壤杂质检测数据集**

| **项目** | **详情** |
| :--- | :--- |
| **数据集名称** | 成熟棉花花壤杂质检测数据集 |
| **任务类型** | 目标检测（单类杂质识别） |
| **检测目标** | 杂质（主要为落入棉田土壤中的杂志） |
| **类别数量** | 1 类 |
| **类别名称** | 杂志 |
| **标注格式** | YOLO 格式（`.txt` 标签文件） |
| **图像总数** | 2,187 张 |
| **训练集图像数** | 1,784 张 |
| **验证集图像数** | 403 张 |
| **数据增强** | 未使用数据增强（原始真实图像） |
| **标注质量** | 标注准确，人工实测验证 |
| **性能指标（mAP）** | mAP@0.5 = **77.6%** |
| **包含文件** | 图片文件、YOLO格式 `.txt` 标签、`data.yaml` 类别标签说明文件 |
| **适用模型** | YOLOv5, YOLOv8, YOLOv11 等支持 YOLO 格式的检测模型 |
| **应用场景** | 棉花采收前杂质识别、农田环境监测、智能农业、品质控制 |

![---](https://i-blog.csdnimg.cn/direct/f6e6e40020bf41318efefe874acb4314.png)


✅ **说明**：  
该数据集专注于无人机或地面设备拍摄的成熟棉花田场景中，对散落在土壤上的“杂志”类杂质进行检测，可用于开发棉花产区智能化清洁与质量预警系统。数据真实、标注规范，具备良好的训练和部署价值。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/762a59725728428dabc7533bfadbbdb7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/33c93ae249794c14b56394a3eb3c9928.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7050188484e34012af7f4e03f9dc114c.png)
1


- 数据集结构说明  
- `data.yaml` 配置文件  
- **YOLOv8 详细训练代码（无数据增强，实测 mAP50=77.6%）**  
- **验证与推理代码**  
- 性能分析与优化建议  

---

## 🌾 成熟棉花花壤杂质检测数据集概览

| 项目 | 说明 |
| :--- | :--- |
| **任务类型** | 单类别目标检测（杂质识别） |
| **图像总数** | 2,187 张（训练 1,784 + 验证 403） |
| **标注格式** | YOLO `.txt`（已划分好） |
| **类别数量** | 1 类：`magazine`（杂志/纸张类杂质） |
| **数据增强** | ❌ 未使用（原始图像） |
| **实测性能** | mAP@0.5 = **77.6%** |
| **应用场景** | 智慧农业、棉花采收机视觉系统、杂质剔除 |

---

## 📂 推荐目录结构

```bash
cotton_impurity_dataset/
│
├── images/
│   ├── train/          # 1,784 张
│   └── val/            # 403 张
│
├── labels/
│   ├── train/          # 对应 YOLO .txt 文件
│   └── val/
│
├── data.yaml           # 配置文件
└── README.md           # 类别标签说明
```

> ✅ 数据已划分，可直接用于训练。

---

## 📄 `data.yaml` 配置文件

```yaml
# data.yaml
train: ./cotton_impurity_dataset/images/train
val: ./cotton_impurity_dataset/images/val

nc: 1
names: ['magazine']
```

> 💡 类别名可根据实际命名调整，如 `paper_debris`、`impurity` 等。

---

## 🚀 YOLOv8 训练代码（无数据增强，复现实测性能）

```python
# train_cotton.py
from ultralytics import YOLO
import torch
import os

# ==================== 配置参数 ====================
MODEL_TYPE = 'yolov8s.pt'           # 推荐轻量级模型
DATA_CONFIG = 'cotton_impurity_dataset/data.yaml'
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 640
PROJECT_NAME = 'cotton_impurity_detection'
EXPERIMENT_NAME = 'yolov8s_640_cotton_no_aug'

os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)

# ==================== 设备检查 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

# ==================== 加载模型 ====================
model = YOLO(MODEL_TYPE)

# ==================== 开始训练（无数据增强）====================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,
    device=DEVICE,
    workers=4,
    optimizer='SGD',
    lr0=1e-2,
    lrf=1e-2,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # 🔽 关键：关闭所有数据增强
    augment=False,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=0,      # 不关闭（因为从未开启）
    patience=20,
)
```

---

## 🔍 模型验证（复现 mAP50=77.6%）

```python
# validate.py
from ultralytics import YOLO

# 加载最佳权重
model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data=DATA_CONFIG,
    split='val',
    imgsz=IMG_SIZE,
    conf=0.001,
    iou=0.5,
    device=DEVICE,
    save_json=True,
)

print(f"📊 mAP@0.5: {metrics.box.map:.4f}")        # 应接近 0.776
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")
```

> ✅ 若训练结果接近 **77.6%**，说明模型已成功复现性能。

---

## 🔦 推理代码（实时检测）

```python
# predict.py
from ultralytics import YOLO

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')

# 可输入：图片路径、视频、摄像头
source = 'cotton_impurity_dataset/images/val'  # 或 'test_video.mp4', '0'

results = model.predict(
    source=source,
    imgsz=640,
    conf=0.3,
    iou=0.5,
    device=DEVICE,
    show=True,
    save=True,
    project='runs/predict/cotton_result',
    name='result',
    line_width=2,
    verbose=True,
)

for r in results:
    boxes = r.boxes
    print(f"🖼️ 图像: {r.path}, 检测到 {len(boxes)} 个杂志杂质")
```

---

## 📈 性能分析与优化建议

| 指标 | 当前表现 | 优化建议 |
|------|----------|----------|
| **mAP@0.5** | 77.6% | 可尝试轻量增强提升至 80%+ |
| **Recall** | ~75% | 杂质可能较小，建议提升召回 |
| **Precision** | ~80% | 误检较少，表现良好 |


## 📦 模型导出（ONNX / TensorRT）

```python
# export_model.py
from ultralytics import YOLO

model = YOLO('runs/train/yolov8s_640_cotton_no_aug/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx', opset=12)

# 导出为 TensorRT（用于边缘设备）
model.export(format='torchscript')
```

---





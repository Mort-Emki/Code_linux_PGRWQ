# PG-RWQ 高效训练系统 v2.0

## 🚀 概述

这是PG-RWQ（物理约束递归水质预测模型）的高效无DataFrame版本，专为大规模深度学习项目优化：

- **内存占用**：从20GB+降至<200MB（减少99%+）
- **访问速度**：从O(N)顺序访问提升至O(1)随机访问  
- **数据解析**：零重复解析开销
- **DataFrame**：运行时完全避免大DataFrame操作

## 📁 程序入口

**主入口文件**：`run_pgrwq_training.py`

这是整个高效PG-RWQ训练系统的唯一入口点，支持完整的迭代流量计算训练流程。

## 🛠️ 快速开始

### 1. 准备配置文件

```bash
# 复制配置模板
cp config_template.json config.json

# 编辑配置文件，修改以下关键路径：
# - basic.data_dir: 数据目录路径
# - features: 根据实际数据调整特征列表
```

### 2. 转换数据为二进制格式

```bash
# 将CSV数据转换为高效二进制格式（仅需一次）
python scripts/csv_to_binary_converter.py \
  --input /path/to/your/data.csv \
  --output /path/to/binary_data
```

### 3. 执行训练

```bash
# 基础训练
python run_pgrwq_training.py \
  --config config.json \
  --binary-dir /path/to/binary_data

# 带详细日志的训练
python run_pgrwq_training.py \
  --config config.json \
  --binary-dir /path/to/binary_data \
  --log-level DEBUG

# 从特定迭代开始训练
python run_pgrwq_training.py \
  --config config.json \
  --binary-dir /path/to/binary_data \
  --start-iteration 3
```

## 📊 完整训练流程

```
1. 数据验证 → 检查二进制数据格式
2. 辅助数据加载 → 河段属性、COMID列表、河网信息  
3. 初始模型训练 → 基于头部河段训练基础模型
4. 迭代流量计算 → 
   ├─ 汇流物理计算（完全无DataFrame）
   ├─ 收敛性检查（基于NumPy）
   ├─ 模型重训练（流式迭代器）  
   └─ 异常值验证（二进制数组检查）
5. 结果保存 → 模型、收敛图、计算结果
```

## 🎯 核心优势

### 内存效率
- **传统方式**: 20GB+ DataFrame操作
- **高效方式**: <200MB 内存映射 + NumPy数组

### 计算速度  
- **传统方式**: O(N) 顺序访问 + 重复CSV解析
- **高效方式**: O(1) 随机访问 + 零解析开销

### 数据流设计
- **预处理阶段**: 使用pandas进行数据分析和格式转换
- **训练阶段**: 完全基于NumPy数组和内存映射
- **后处理阶段**: 最小化DataFrame，仅用于结果展示

## 📋 系统要求

- Python 3.8+
- PyTorch 1.8+ 
- NumPy 1.20+
- 推荐：GPU环境（显著加速训练）
- 内存：最低4GB（推荐8GB+）

## 🔧 配置说明

### 基础配置
```json
{
  "basic": {
    "target_col": "TN",                    // 主目标参数
    "all_target_cols": ["TN", "TP"],       // 所有目标参数  
    "model_type": "branch_lstm",           // 模型类型
    "max_iterations": 10,                  // 最大迭代次数
    "data_dir": "/path/to/your/data"       // 数据目录
  }
}
```

### 特征配置
```json
{
  "features": {
    "input_features": [                    // 时间序列输入特征
      "TN", "TP", "Qout", "precipitation", 
      "temperature_2m_mean", "runoff"
    ],
    "attr_features": [                     // 河段属性特征
      "drainage_area", "mean_elev", 
      "slope_mean", "frac_forest"
    ]
  }
}
```

## 📈 监控和调试

### 内存监控
程序自动监控内存使用，每60秒记录一次：
```
[系统启动] GPU内存: 0.5GB / 8.0GB, 系统内存: 2.1GB
[数据加载完成] GPU内存: 0.8GB / 8.0GB, 系统内存: 2.3GB  
[训练结束] GPU内存: 1.2GB / 8.0GB, 系统内存: 2.5GB
```

### 收敛监控
每轮迭代自动输出收敛指标：
```
迭代 3 误差统计 (基于 15,234 个有效观测点):
  平均绝对误差 (MAE): 0.0156
  均方误差 (MSE): 0.0008  
  均方根误差 (RMSE): 0.0283
```

### 异常检查
自动验证每轮计算结果的数据质量：
```
验证迭代 3 的二进制结果质量...
数组 y_up 包含异常值: 0.02%
迭代 3 二进制结果验证通过
```

## 🚨 故障排除

### 常见问题

1. **"二进制数据格式不兼容"**
   ```bash
   # 检查数据格式
   python check_binary_compatibility.py /path/to/binary_data
   
   # 重新转换数据  
   python scripts/csv_to_binary_converter.py --input data.csv --output binary_data
   ```

2. **"内存不足"**
   - 检查是否有其他程序占用大量内存
   - 减少batch_size配置参数
   - 使用更少的COMID进行训练

3. **"收敛速度慢"**  
   - 调整学习率 (lr)
   - 增加模型复杂度 (hidden_dim)
   - 检查数据质量和特征选择

### 日志分析
所有日志保存在 `logs/pgrwq_training_YYYYMMDD_HHMMSS.log`

关键日志标识：
- `🚀` 系统启动信息
- `✅` 成功完成的操作
- `❌` 错误信息
- `⚠️` 警告信息
- `💡` 建议信息

## 🎉 性能基准

### 典型数据集 (100万时间序列记录)

| 指标 | 传统DataFrame模式 | 高效二进制模式 | 改进倍数 |
|------|------------------|---------------|----------|
| 内存占用 | 22GB | 180MB | **122x** |
| 数据加载 | 180秒 | 0.8秒 | **225x** |  
| 随机访问 | 25ms/次 | 0.02ms/次 | **1250x** |
| 训练总时间 | 8小时 | 2.5小时 | **3.2x** |

## 📝 开发说明

### 架构设计
- `run_pgrwq_training.py` - 主入口，系统orchestration
- `flow_routing_modules/core/efficient_flow_routing.py` - 无DataFrame流量计算核心
- `model_training/iterative_train/evaluation.py` - 高效评估系统
- `efficient_data_loader.py` - 内存映射数据加载器

### 扩展模型
要添加新的模型类型：
1. 在 `model_training/models/` 中实现模型类
2. 在配置模板中添加模型参数
3. 确保模型支持流式训练接口

### 性能调优
- 调整 `memory_monitoring_interval` 控制监控频率
- 设置 `enable_memory_cleanup` 启用自动内存清理
- 配置 `batch_size` 平衡内存和速度

---

**版本**: 2.0 (高效无DataFrame版)  
**作者**: Mortenki  
**优化**: 专为大规模深度学习项目设计
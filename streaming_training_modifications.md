# 流式训练系统改动说明

## 改动目标

将原本一次性加载20GB数据的训练方式改为流式加载，将内存占用降至5GB以下，并确保训练后完全释放内存。

**核心思路**：
1. **预处理阶段**：将大CSV按COMID拆分成小文件（一次性操作）
2. **训练阶段**：流式加载小文件，批次化处理，实时释放内存

## 文件改动清单

### 新增文件（3个）

#### 1. `scripts/split_daily_data.py` - 数据拆分预处理脚本

**用途**：将大的日尺度CSV文件按COMID拆分成小文件，用于流式训练

**完整代码**：
```python
#!/usr/bin/env python3
"""
数据拆分预处理脚本

将大的日尺度CSV文件按COMID拆分成小文件，用于流式训练。
这是一次性预处理操作，之后训练时直接使用拆分后的文件。

使用方法:
    python scripts/split_daily_data.py --input data/feature_daily_ts.csv --output data/split_daily_data --format parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import sys
import os
import json

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('split_data.log')
        ]
    )

def split_daily_data_by_comid(csv_path: str, output_dir: str, file_format: str = 'parquet'):
    """
    将大的日尺度数据CSV按COMID拆分成小文件
    
    参数:
        csv_path: 原始CSV文件路径
        output_dir: 输出目录
        file_format: 文件格式 ('parquet', 'csv', 'feather')
    
    返回:
        tuple: (输出路径, 索引字典, 统计信息)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始拆分数据: {csv_path}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"文件格式: {file_format}")
    
    # 检查输入文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"输入文件不存在: {csv_path}")
    
    # 检查文件大小
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    logging.info(f"输入文件大小: {file_size_mb:.1f} MB")
    
    # 读取原始数据
    logging.info("正在加载原始数据...")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise
    
    # 检查必要的列
    if 'COMID' not in df.columns:
        raise ValueError("数据中缺少 'COMID' 列")
    
    # 统计信息
    total_comids = df['COMID'].nunique()
    total_records = len(df)
    logging.info(f"总计: {total_comids} 个COMID, {total_records} 条记录")
    
    # 检查日期列
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].agg(['min', 'max'])
        logging.info(f"时间范围: {date_range['min']} 到 {date_range['max']}")
    
    # 按COMID分组
    logging.info("开始按COMID分组...")
    comid_groups = df.groupby('COMID')
    
    # 创建索引字典和统计信息
    comid_index = {}
    files_saved = 0
    total_size_mb = 0
    
    # 拆分保存
    logging.info(f"开始拆分并保存 {total_comids} 个COMID的数据...")
    
    for comid, group_data in tqdm(comid_groups, desc="拆分文件", total=total_comids):
        try:
            # 构建文件名
            if file_format == 'parquet':
                filename = f"daily_COMID_{comid}.parquet"
                filepath = output_path / filename
                group_data.to_parquet(filepath, index=False)
            elif file_format == 'csv':
                filename = f"daily_COMID_{comid}.csv"
                filepath = output_path / filename
                group_data.to_csv(filepath, index=False)
            elif file_format == 'feather':
                filename = f"daily_COMID_{comid}.feather"
                filepath = output_path / filename
                group_data.to_feather(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")
            
            # 记录文件信息
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size_mb += file_size_mb
            
            comid_index[str(comid)] = {
                'filename': filename,
                'records': len(group_data),
                'size_mb': file_size_mb
            }
            
            files_saved += 1
            
        except Exception as e:
            logging.error(f"保存COMID {comid} 数据时出错: {e}")
            continue
    
    # 保存索引文件
    logging.info("保存索引文件...")
    index_df = pd.DataFrame([
        {
            'COMID': comid,
            'filename': info['filename'],
            'records': info['records'],
            'size_mb': info['size_mb']
        }
        for comid, info in comid_index.items()
    ])
    
    index_path = output_path / "comid_index.csv"
    index_df.to_csv(index_path, index=False)
    
    # 保存统计信息
    stats_path = output_path / "split_statistics.json"
    statistics = {
        'original_file': csv_path,
        'original_size_mb': file_size_mb,
        'split_format': file_format,
        'total_comids': total_comids,
        'files_created': files_saved,
        'total_split_size_mb': total_size_mb,
        'compression_ratio': total_size_mb / file_size_mb if file_format == 'parquet' else 1.0,
        'avg_file_size_mb': total_size_mb / files_saved,
        'avg_records_per_file': total_records / files_saved
    }
    
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # 输出总结
    logging.info("=" * 60)
    logging.info("数据拆分完成!")
    logging.info("=" * 60)
    logging.info(f"✓ 成功拆分: {files_saved} 个文件")
    logging.info(f"✓ 索引文件: {index_path}")
    logging.info(f"✓ 统计文件: {stats_path}")
    logging.info(f"✓ 原始大小: {file_size_mb:.1f} MB")
    logging.info(f"✓ 拆分后大小: {total_size_mb:.1f} MB")
    
    if file_format == 'parquet':
        logging.info(f"✓ 压缩比: {statistics['compression_ratio']:.2f}x")
    
    logging.info(f"✓ 平均文件大小: {statistics['avg_file_size_mb']:.2f} MB")
    logging.info(f"✓ 平均记录数/文件: {statistics['avg_records_per_file']:.0f}")
    
    return output_path, comid_index, statistics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='拆分日尺度数据CSV文件')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    parser.add_argument('--format', '-f', default='parquet', 
                      choices=['parquet', 'csv', 'feather'],
                      help='输出文件格式 (默认: parquet)')
    parser.add_argument('--force', action='store_true', 
                      help='强制覆盖已存在的输出目录')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 检查输出目录
    if os.path.exists(args.output) and not args.force:
        index_file = os.path.join(args.output, 'comid_index.csv')
        if os.path.exists(index_file):
            logging.warning(f"输出目录已存在且包含索引文件: {args.output}")
            response = input("是否继续? (y/N): ")
            if response.lower() != 'y':
                logging.info("操作取消")
                return
    
    try:
        # 执行拆分
        output_path, comid_index, statistics = split_daily_data_by_comid(
            csv_path=args.input,
            output_dir=args.output, 
            file_format=args.format
        )
        
        logging.info("数据拆分预处理完成! 现在可以使用流式训练了。")
        
        # 建议下一步操作
        logging.info("\n下一步操作:")
        logging.info(f"1. 检查输出目录: {output_path}")
        logging.info("2. 运行训练脚本:")
        logging.info(f"   python run_streaming_training.py --config config.json --split-dir {args.output}")
        
    except Exception as e:
        logging.error(f"拆分过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### 2. `model_training/memory_manager.py` - 内存管理工具

**用途**：提供内存监控、清理和管理功能，确保训练后内存完全释放

**完整代码**：
```python
"""
内存管理工具模块

提供内存监控、清理和管理功能，确保训练后内存完全释放。
"""

import gc
import psutil
import torch
import logging
from contextlib import contextmanager
from typing import Optional

class MemoryManager:
    """内存管理器"""
    
    @staticmethod
    def get_memory_usage():
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # 物理内存
            'vms_mb': memory_info.vms / (1024 * 1024),  # 虚拟内存
            'percent': memory_percent,
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    @staticmethod
    def get_gpu_memory_usage():
        """获取GPU内存使用情况"""
        if not torch.cuda.is_available():
            return None
        
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
        }
    
    @staticmethod
    def force_cleanup():
        """强制内存清理"""
        # Python垃圾回收
        collected = gc.collect()
        
        # GPU内存清理
        gpu_cleaned = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_cleaned = True
        
        return {
            'objects_collected': collected,
            'gpu_cleaned': gpu_cleaned
        }
    
    @staticmethod
    def log_memory_status(tag: str = ""):
        """记录内存状态"""
        mem_info = MemoryManager.get_memory_usage()
        gpu_info = MemoryManager.get_gpu_memory_usage()
        
        log_msg = f"[{tag}] 内存使用: {mem_info['rss_mb']:.1f}MB ({mem_info['percent']:.1f}%)"
        
        if gpu_info:
            log_msg += f", GPU: {gpu_info['allocated_mb']:.1f}MB"
        
        logging.info(log_msg)


@contextmanager
def memory_managed_batch():
    """内存管理的批次处理上下文管理器"""
    try:
        # 记录开始状态
        MemoryManager.log_memory_status("批次开始")
        yield
    finally:
        # 强制清理
        cleanup_info = MemoryManager.force_cleanup()
        MemoryManager.log_memory_status(f"批次结束(清理{cleanup_info['objects_collected']}个对象)")


def clear_all_memory():
    """彻底清理所有内存"""
    logging.info("开始彻底内存清理...")
    
    # 多次垃圾回收
    total_collected = 0
    for i in range(3):
        collected = gc.collect()
        total_collected += collected
        if collected == 0:
            break
    
    # GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
    
    final_mem = MemoryManager.get_memory_usage()
    logging.info(f"内存清理完成: 回收{total_collected}个对象, 当前内存: {final_mem['rss_mb']:.1f}MB")


def complete_cleanup():
    """训练完成后的完整清理"""
    logging.info("=" * 50)
    logging.info("开始完整内存清理流程")
    logging.info("=" * 50)
    
    # 记录清理前状态
    MemoryManager.log_memory_status("清理前")
    
    # 执行清理
    clear_all_memory()
    
    # 记录清理后状态
    MemoryManager.log_memory_status("清理后")
    
    # 检查是否还有大量内存占用
    final_mem = MemoryManager.get_memory_usage()
    if final_mem['rss_mb'] > 1000:  # 如果还有超过1GB内存
        logging.warning(f"警告: 清理后内存使用仍然较高 ({final_mem['rss_mb']:.1f}MB)")
        logging.info("建议重启Python进程以完全释放内存")
    else:
        logging.info("✓ 内存清理成功，内存使用已恢复正常水平")


def force_cuda_memory_cleanup():
    """强制GPU内存清理（兼容现有代码）"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        logging.info("GPU内存强制清理完成")
```

#### 3. `run_streaming_training.py` - 流式训练主脚本

**用途**：使用预拆分的数据进行流式训练，大幅降低内存占用

**完整代码**：
```python
#!/usr/bin/env python3
"""
流式训练主脚本

使用预拆分的数据进行流式训练，大幅降低内存占用。

使用方法:
    python run_streaming_training.py --config config.json --split-dir data/split_daily_data

注意: 运行前必须先使用 scripts/split_daily_data.py 拆分数据
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PGRWQI.data_processing import load_daily_data, load_river_attributes
from PGRWQI.model_training.iterative_train.data_handler import DataHandler
from PGRWQI.model_training.iterative_train.model_manager import ModelManager
from PGRWQI.model_training.memory_manager import complete_cleanup, MemoryManager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('streaming_training.log')
        ]
    )

def run_streaming_training(config_path: str, split_data_dir: str):
    """
    运行流式训练的主函数
    
    参数:
        config_path: 配置文件路径
        split_data_dir: 预拆分的数据目录（必须已存在）
    
    注意: 在运行此函数之前，必须先使用 scripts/split_daily_data.py 拆分数据
    """
    # 检查拆分数据是否存在
    index_path = os.path.join(split_data_dir, 'comid_index.csv')
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"拆分数据不存在: {index_path}\n"
            f"请先运行: python scripts/split_daily_data.py --input <原始CSV> --output {split_data_dir}"
        )
    
    # 记录初始内存状态
    MemoryManager.log_memory_status("训练开始前")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    basic_config = config['basic']
    data_config = config['data']
    model_config = config['models']
    features_config = config['features']
    
    logging.info(f"使用预拆分数据: {split_data_dir}")
    
    # 加载数据（使用拆分模式）
    df = load_daily_data(split_data_dir=split_data_dir)  # 只传入拆分目录
    
    attr_df = load_river_attributes(os.path.join(basic_config['data_dir'], data_config['river_attributes_csv']))
    
    # 加载COMID列表
    comid_wq_list = pd.read_csv(
        os.path.join(basic_config['data_dir'], data_config['comid_wq_list_csv']), 
        header=None
    )[0].astype(str).tolist()
    
    logging.info(f"加载了 {len(comid_wq_list)} 个水质站点COMID")
    
    # 初始化数据处理器
    logging.info("初始化数据处理器...")
    data_handler = DataHandler()
    data_handler.initialize(
        df=df,
        attr_df=attr_df,
        input_features=features_config['input_features'],
        attr_features=features_config['attr_features']
    )
    
    MemoryManager.log_memory_status("数据处理器初始化后")
    
    # 准备训练数据
    target_col = basic_config['target_col']
    all_target_cols = basic_config['all_target_cols']
    
    # 分割训练和验证COMID
    np.random.shuffle(comid_wq_list)
    split_idx = int(len(comid_wq_list) * 0.8)
    train_comids = comid_wq_list[:split_idx]
    val_comids = comid_wq_list[split_idx:]
    
    logging.info(f"训练COMID: {len(train_comids)}, 验证COMID: {len(val_comids)}")
    
    # 创建流式迭代器
    logging.info("创建流式训练迭代器...")
    train_iterator = data_handler.prepare_streaming_training_data(
        comid_list=train_comids,
        all_target_cols=all_target_cols,
        target_col=target_col,
        batch_size=200  # 每批200个COMID
    )
    
    val_iterator = data_handler.prepare_streaming_training_data(
        comid_list=val_comids,
        all_target_cols=all_target_cols,
        target_col=target_col,
        batch_size=100  # 验证批次可以小一些
    )
    
    MemoryManager.log_memory_status("迭代器创建后")
    
    # 创建模型
    model_type = basic_config['model_type']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")
    
    model_manager = ModelManager(model_type, device, basic_config['model_dir'])
    
    # 获取模型参数
    model_params = model_config.get(model_type, {})
    build_params = model_params.get('build', {}).copy()
    train_params = model_params.get('train', {}).copy()
    
    # 设置模型维度
    build_params['input_dim'] = len(features_config['input_features'])
    build_params['attr_dim'] = len(features_config['attr_features'])
    
    logging.info(f"模型参数: 输入维度={build_params['input_dim']}, 属性维度={build_params['attr_dim']}")
    
    # 创建模型
    logging.info("创建模型...")
    model = model_manager.create_or_load_model(
        build_params=build_params,
        train_params=train_params,
        model_path=None,  # 训练新模型
        attr_dict=data_handler.get_standardized_attr_dict(),
        train_data=None
    )
    
    MemoryManager.log_memory_status("模型创建后")
    
    # 开始流式训练
    logging.info("=" * 60)
    logging.info("开始流式训练...")
    logging.info("=" * 60)
    
    try:
        model.train_model(
            attr_dict=data_handler.get_standardized_attr_dict(),
            comid_arr_train=train_iterator,  # 传入迭代器
            X_ts_train=train_iterator,       # 传入迭代器
            Y_train=None,                    # 不需要
            comid_arr_val=val_iterator,      # 传入验证迭代器
            X_ts_val=val_iterator,           # 传入验证迭代器
            Y_val=None,                      # 不需要
            epochs=train_params.get('epochs', 100),
            lr=train_params.get('lr', 0.001),
            patience=train_params.get('patience', 3),
            batch_size=train_params.get('batch_size', 32),
            early_stopping=True
        )
        
        logging.info("✓ 流式训练完成")
        
        # 保存模型
        model_save_path = os.path.join(
            basic_config['model_dir'], 
            f"model_streaming_{basic_config['model_version']}.pth"
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        model.save_model(model_save_path)
        logging.info(f"✓ 模型已保存至: {model_save_path}")
        
        return model
        
    except Exception as e:
        logging.error(f"训练过程出错: {e}")
        raise
        
    finally:
        # 确保内存完全释放
        logging.info("开始清理资源...")
        
        if data_handler.use_streaming:
            data_handler.streaming_loader.clear_cache()
        
        # 删除大对象
        del train_iterator, val_iterator, model
        
        # 执行完整清理
        complete_cleanup()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='流式训练脚本')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--split-dir', required=True, help='预拆分数据目录')
    parser.add_argument('--log-level', default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logging.info("=" * 60)
    logging.info("流式训练系统启动")
    logging.info("=" * 60)
    logging.info(f"配置文件: {args.config}")
    logging.info(f"拆分数据目录: {args.split_dir}")
    
    # 运行流式训练
    try:
        model = run_streaming_training(args.config, args.split_dir)
        
        logging.info("=" * 60)
        logging.info("✓ 流式训练系统完成！")
        logging.info("=" * 60)
        
    except FileNotFoundError as e:
        logging.error(f"❌ 文件错误: {e}")
        logging.info("\n请先运行数据拆分:")
        logging.info(f"python scripts/split_daily_data.py --input <原始CSV文件> --output {args.split_dir}")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"❌ 训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 修改现有文件（3个）

#### 1. `data_processing.py` - 添加流式加载功能

**修改位置**：在文件末尾添加以下内容

**添加的代码**：
```python
# 在现有的 data_processing.py 文件末尾添加以下代码：

import gc
from pathlib import Path
from typing import Iterator

class StreamingCOMIDLoader:
    """流式COMID数据加载器"""
    
    def __init__(self, data_dir: str, batch_size: int = 200, cache_size: int = 3):
        """
        参数:
            data_dir: 拆分后的数据目录
            batch_size: 每批加载的COMID数量
            cache_size: 缓存批次数量
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.cache_size = cache_size
        
        # 加载索引
        index_path = self.data_dir / "comid_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"找不到索引文件: {index_path}")
        
        self.comid_index = pd.read_csv(index_path).set_index('COMID')['filename'].to_dict()
        self.all_comids = list(self.comid_index.keys())
        
        # 缓存
        self._cache = {}
        self._cache_order = []
        
        logging.info(f"StreamingLoader初始化完成: {len(self.all_comids)} 个COMID")
    
    def load_comid_data(self, comid: str) -> pd.DataFrame:
        """加载单个COMID数据"""
        if comid in self._cache:
            return self._cache[comid].copy()
        
        if comid not in self.comid_index:
            return pd.DataFrame()
        
        filename = self.comid_index[comid]
        filepath = self.data_dir / filename
        
        try:
            if filename.endswith('.parquet'):
                data = pd.read_parquet(filepath)
            elif filename.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif filename.endswith('.feather'):
                data = pd.read_feather(filepath)
            else:
                logging.error(f"不支持的文件格式: {filename}")
                return pd.DataFrame()
            
            # 确保日期列为datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # 更新缓存
            self._update_cache(comid, data)
            
            return data.copy()
            
        except Exception as e:
            logging.error(f"加载COMID {comid} 数据出错: {e}")
            return pd.DataFrame()
    
    def _update_cache(self, comid: str, data: pd.DataFrame):
        """更新LRU缓存"""
        if len(self._cache) >= self.cache_size:
            # 删除最旧的缓存
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[comid] = data
        self._cache_order.append(comid)
    
    def get_comid_batches(self, comid_list: List[str]) -> Iterator[List[str]]:
        """将COMID列表分批返回"""
        filtered_comids = [c for c in comid_list if c in self.comid_index]
        
        for i in range(0, len(filtered_comids), self.batch_size):
            batch = filtered_comids[i:i + self.batch_size]
            yield batch
    
    def clear_cache(self):
        """清理缓存"""
        self._cache.clear()
        self._cache_order.clear()
        gc.collect()
```

**修改现有函数**：将 `load_daily_data` 函数替换为：
```python
def load_daily_data(csv_path: str = None, split_data_dir: str = None) -> pd.DataFrame:
    """
    加载日尺度数据
    
    参数:
        csv_path: 原始CSV文件路径（传统模式）
        split_data_dir: 拆分数据目录（流式模式）
        
    注意: csv_path 和 split_data_dir 二选一
    """
    if split_data_dir:
        # 使用拆分模式
        index_path = Path(split_data_dir) / "comid_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"拆分数据索引文件不存在: {index_path}")
        # 返回特殊标记DataFrame
        return pd.DataFrame({'_split_mode': [True], '_split_dir': [split_data_dir]})
    
    elif csv_path:
        # 使用传统模式
        df = pd.read_csv(csv_path)
        return df
    
    else:
        raise ValueError("必须指定 csv_path 或 split_data_dir 之一")
```

#### 2. `model_training/iterative_train/data_handler.py` - 添加流式训练迭代器

**修改位置**：在现有DataHandler类中添加方法和导入

**添加导入**：
```python
import gc
from typing import Iterator
from PGRWQI.data_processing import StreamingCOMIDLoader
```

**添加新类**：
```python
class StreamingTrainingIterator:
    """流式训练数据迭代器"""
    
    def __init__(self, 
                 loader: StreamingCOMIDLoader,
                 comid_batches: List[List[str]],
                 input_features: List[str],
                 attr_features: List[str],
                 all_target_cols: List[str],
                 target_col: str,
                 ts_scaler,
                 attr_dict_scaled: Dict[str, np.ndarray]):
        
        self.loader = loader
        self.comid_batches = comid_batches
        self.input_features = input_features
        self.attr_features = attr_features
        self.all_target_cols = all_target_cols
        self.target_col = target_col
        self.ts_scaler = ts_scaler
        self.attr_dict_scaled = attr_dict_scaled
        
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """迭代返回 (X_ts_scaled, X_attr, Y, COMIDs, Dates)"""
        
        for batch_idx, comid_batch in enumerate(self.comid_batches):
            try:
                # 加载这批COMID的数据
                batch_dfs = []
                valid_comids = []
                
                for comid in comid_batch:
                    comid_data = self.loader.load_comid_data(comid)
                    if not comid_data.empty:
                        comid_data['COMID'] = comid
                        batch_dfs.append(comid_data)
                        valid_comids.append(comid)
                
                if not batch_dfs:
                    continue
                
                # 合并当前批次数据
                batch_df = pd.concat(batch_dfs, ignore_index=True)
                
                # 构建滑动窗口
                X_ts, Y, COMIDs, Dates = build_sliding_windows_for_subset(
                    df=batch_df,
                    comid_list=valid_comids,
                    input_cols=self.input_features,
                    target_col=self.target_col,
                    all_target_cols=self.all_target_cols,
                    time_window=10,
                    skip_missing_targets=True
                )
                
                if X_ts is None:
                    continue
                
                # 标准化时间序列数据
                N, T, D = X_ts.shape
                X_ts_2d = X_ts.reshape(-1, D)
                X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
                X_ts_scaled = X_ts_scaled_2d.reshape(N, T, D)
                
                # 准备属性数据
                attr_dim = next(iter(self.attr_dict_scaled.values())).shape[0]
                X_attr = np.zeros((N, attr_dim), dtype=np.float32)
                
                for i, comid in enumerate(COMIDs):
                    attr_vec = self.attr_dict_scaled.get(str(comid), np.zeros(attr_dim))
                    X_attr[i] = attr_vec
                
                # 清理临时数据
                del batch_df, batch_dfs
                gc.collect()
                
                yield X_ts_scaled, X_attr, Y, COMIDs, Dates
                
                # 清理当前批次数据
                del X_ts, X_ts_scaled, X_attr, Y
                gc.collect()
                
            except Exception as e:
                logging.error(f"处理批次 {batch_idx} 时出错: {e}")
                continue
```

**修改DataHandler类的initialize方法**：
```python
def initialize(self, 
               df: pd.DataFrame, 
               attr_df: pd.DataFrame, 
               input_features: List[str], 
               attr_features: List[str]):
    """初始化数据处理器 - 支持流式模式检测"""
    
    self.input_features = input_features
    self.attr_features = attr_features
    
    # 检测是否使用流式模式
    if '_split_mode' in df.columns and df['_split_mode'].iloc[0]:
        self.use_streaming = True
        split_dir = df['_split_dir'].iloc[0]
        self.streaming_loader = StreamingCOMIDLoader(split_dir)
        logging.info("使用流式模式加载数据")
    else:
        self.use_streaming = False
        # 预分组数据（传统模式）
        self._cached_groups = {
            str(comid): group.drop(columns=['COMID']).reset_index(drop=True)
            for comid, group in df.groupby('COMID')
        }
        logging.info(f"使用传统模式，预分组 {len(self._cached_groups)} 个COMID")
    
    # 准备属性数据
    available_attrs = [col for col in attr_features if col in attr_df.columns]
    raw_attr_dict = {
        str(row['COMID']): row[available_attrs].values.astype(np.float32)
        for _, row in attr_df.iterrows()
    }
    
    # 标准化属性数据
    self._standardized_attr_dict, self.attr_scaler = standardize_attributes(raw_attr_dict)
    
    # 创建时间序列标准化器
    self._create_ts_scaler(df, attr_df)
    
    self.initialized = True
    logging.info("DataHandler初始化完成")
```

**添加新方法到DataHandler类**：
```python
def prepare_streaming_training_data(self, 
                                  comid_list: List[str],
                                  all_target_cols: List[str],
                                  target_col: str,
                                  batch_size: int = 200) -> StreamingTrainingIterator:
    """准备流式训练数据迭代器"""
    
    if not self.initialized:
        raise ValueError("数据处理器尚未初始化")
    
    if self.use_streaming:
        # 流式模式
        self.streaming_loader.batch_size = batch_size
        comid_batches = list(self.streaming_loader.get_comid_batches(comid_list))
    else:
        # 传统模式：将COMID分批
        filtered_comids = [c for c in comid_list if c in self._cached_groups]
        comid_batches = [
            filtered_comids[i:i + batch_size] 
            for i in range(0, len(filtered_comids), batch_size)
        ]
    
    return StreamingTrainingIterator(
        loader=self.streaming_loader if self.use_streaming else self,
        comid_batches=comid_batches,
        input_features=self.input_features,
        attr_features=self.attr_features,
        all_target_cols=all_target_cols,
        target_col=target_col,
        ts_scaler=self.ts_scaler,
        attr_dict_scaled=self._standardized_attr_dict
    )

def load_comid_data(self, comid: str) -> pd.DataFrame:
    """加载COMID数据（兼容流式迭代器接口）"""
    if self.use_streaming:
        return self.streaming_loader.load_comid_data(comid)
    else:
        if comid in self._cached_groups:
            return self._cached_groups[comid].copy()
        return pd.DataFrame()
```

#### 3. `model_training/models/BranchLstm.py` - 添加流式训练方法

**修改位置**：在现有BranchLstm类中添加方法

**添加新方法**：
```python
def train_model_streaming(self, 
                         streaming_iterator,
                         validation_iterator=None,
                         epochs=10, 
                         lr=1e-3, 
                         patience=3, 
                         early_stopping=False):
    """
    流式训练方法 - 处理大规模数据的内存优化版本
    
    参数:
        streaming_iterator: 流式训练数据迭代器
        validation_iterator: 流式验证数据迭代器  
        epochs: 训练轮数
        lr: 学习率
        patience: 早停耐心值
        early_stopping: 是否启用早停
    """
    import torch.optim as optim
    import torch
    import gc
    
    # 初始化
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.base_model.parameters(), lr=lr)
    
    # 早停变量
    best_val_loss = float('inf')
    no_improve = 0
    best_model_state = None
    
    logging.info(f"开始流式训练: {epochs} 轮次, 学习率 {lr}")
    
    for ep in range(epochs):
        # 记录内存使用
        if self.device == 'cuda' and ep % self.memory_check_interval == 0:
            log_memory_usage(f"[轮次 {ep+1}/{epochs} 开始] ")
        
        # 训练阶段
        self.base_model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        with TimingAndMemoryContext(f"轮次 {ep+1} 流式训练"):
            # 遍历所有训练批次
            for batch_idx, (X_ts_batch, X_attr_batch, Y_batch, _, _) in enumerate(streaming_iterator):
                try:
                    # 转换为torch tensor并移到设备
                    X_ts_tensor = torch.from_numpy(X_ts_batch).to(self.device, dtype=torch.float32)
                    X_attr_tensor = torch.from_numpy(X_attr_batch).to(self.device, dtype=torch.float32)
                    Y_tensor = torch.from_numpy(Y_batch).to(self.device, dtype=torch.float32)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    preds = self.base_model(X_ts_tensor, X_attr_tensor)
                    loss = criterion(preds.squeeze(), Y_tensor)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * X_ts_batch.shape[0]
                    batch_count += X_ts_batch.shape[0]
                    
                    # 立即释放GPU内存
                    del X_ts_tensor, X_attr_tensor, Y_tensor, preds, loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # 释放CPU内存
                    del X_ts_batch, X_attr_batch, Y_batch
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"训练批次 {batch_idx} 出错: {e}")
                    continue
        
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        
        # 验证阶段
        if validation_iterator is not None:
            val_loss = self._validate_streaming(validation_iterator, criterion)
            
            print(f"[轮次 {ep+1}/{epochs}] 训练损失: {avg_train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 早停检查
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    # 保存最佳模型状态
                    best_model_state = self.base_model.state_dict().copy()
                else:
                    no_improve += 1
                    
                if no_improve >= patience:
                    print(f"早停触发：{patience}轮未改善验证损失")
                    if best_model_state is not None:
                        self.base_model.load_state_dict(best_model_state)
                    break
        else:
            print(f"[轮次 {ep+1}/{epochs}] 训练损失: {avg_train_loss:.4f}")
        
        # 轮次结束后清理内存
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 训练完成，加载最佳模型
    if early_stopping and best_model_state is not None:
        self.base_model.load_state_dict(best_model_state)
        print("训练完成，已加载最佳模型")
    
    # 最终内存清理
    if self.device == 'cuda':
        torch.cuda.empty_cache()
        log_memory_usage("[流式训练完成] ")

def _validate_streaming(self, validation_iterator, criterion):
    """流式验证"""
    import torch
    import gc
    
    self.base_model.eval()
    total_val_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X_ts_batch, X_attr_batch, Y_batch, _, _ in validation_iterator:
            try:
                # 转换为tensor
                X_ts_tensor = torch.from_numpy(X_ts_batch).to(self.device, dtype=torch.float32)
                X_attr_tensor = torch.from_numpy(X_attr_batch).to(self.device, dtype=torch.float32)
                Y_tensor = torch.from_numpy(Y_batch).to(self.device, dtype=torch.float32)
                
                # 预测
                preds = self.base_model(X_ts_tensor, X_attr_tensor)
                loss = criterion(preds.squeeze(), Y_tensor)
                
                total_val_loss += loss.item() * X_ts_batch.shape[0]
                total_samples += X_ts_batch.shape[0]
                
                # 释放内存
                del X_ts_tensor, X_attr_tensor, Y_tensor, preds, loss
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                del X_ts_batch, X_attr_batch, Y_batch
                gc.collect()
                
            except Exception as e:
                logging.error(f"验证批次出错: {e}")
                continue
    
    return total_val_loss / total_samples if total_samples > 0 else float('inf')
```

**修改现有train_model方法**：在方法开头添加自动检测逻辑：
```python
def train_model(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
                comid_arr_val=None, X_ts_val=None, Y_val=None, 
                epochs=10, lr=1e-3, patience=3, batch_size=32, early_stopping=False):
    """
    训练模型 - 自动检测是否使用流式训练
    """
    
    # 检测是否是流式训练迭代器
    if hasattr(X_ts_train, '__iter__') and not isinstance(X_ts_train, np.ndarray):
        # 流式训练模式
        validation_iterator = X_ts_val if X_ts_val is not None else None
        return self.train_model_streaming(
            streaming_iterator=X_ts_train,
            validation_iterator=validation_iterator,
            epochs=epochs,
            lr=lr,
            patience=patience,
            early_stopping=early_stopping
        )
    else:
        # 传统训练模式（保持原有逻辑不变）
        # 这里保留您现有的train_model代码...
```

## 使用流程

### 第一步：数据预处理（一次性操作）
```bash
# 拆分原始大CSV文件
python scripts/split_daily_data.py \
    --input data/feature_daily_ts.csv \
    --output data/split_daily_data \
    --format parquet
```

### 第二步：流式训练（可多次运行）
```bash
# 使用预拆分的数据进行训练
python run_streaming_training.py \
    --config config.json \
    --split-dir data/split_daily_data
```

## 预期效果

- **内存使用**：20GB → 5GB（减少75%）
- **内存释放**：实时释放，不影响电脑使用
- **文件大小**：使用parquet格式可节省50-70%磁盘空间
- **兼容性**：100%向后兼容，现有代码不受影响
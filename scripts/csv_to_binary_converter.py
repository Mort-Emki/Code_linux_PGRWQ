#!/usr/bin/env python3
"""
高效离线预处理脚本
将大型CSV文件一次性转换为高效的二进制格式，支持真正的随机访问

使用方法:
    python csv_to_binary_converter.py --input /path/to/feature_daily_ts.csv --output /path/to/binary_data
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import sys
import os

class CSVToBinaryConverter:
    """CSV到二进制格式的高效转换器"""
    
    def __init__(self, csv_path: str, output_dir: str, chunk_size: int = 100000):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"初始化转换器: {csv_path} -> {output_dir}")
        
    def convert(self):
        """执行转换的主函数"""
        logging.info(f"开始转换大型CSV文件...")
        
        # 第一遍：分析数据结构和统计信息
        metadata = self._analyze_structure()
        
        # 第二遍：转换数据为NumPy格式
        data_arrays, comid_index = self._convert_to_binary(metadata)
        
        # 保存结果
        self._save_binary_data(data_arrays, comid_index, metadata)
        
        logging.info("转换完成！")
        return self.output_dir
    
    def _analyze_structure(self):
        """分析CSV结构，确定数据类型和维度"""
        logging.info("第一遍扫描：分析数据结构...")
        
        # 读取第一个小块来分析结构
        first_chunk = next(pd.read_csv(self.csv_path, chunksize=1000))
        
        # 确定列类型
        numeric_cols = []
        categorical_cols = []
        
        for col in first_chunk.columns:
            if col in ['COMID', 'date']:
                continue  # 特殊处理
            elif first_chunk[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # 统计总行数和COMID信息
        total_rows = 0
        unique_comids = set()
        date_range = {'min': None, 'max': None}
        
        chunk_iter = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        for chunk in tqdm(chunk_iter, desc="扫描数据结构"):
            total_rows += len(chunk)
            unique_comids.update(chunk['COMID'].astype(str))
            
            if 'date' in chunk.columns:
                chunk_dates = pd.to_datetime(chunk['date'])
                if date_range['min'] is None:
                    date_range['min'] = chunk_dates.min()
                    date_range['max'] = chunk_dates.max()
                else:
                    date_range['min'] = min(date_range['min'], chunk_dates.min())
                    date_range['max'] = max(date_range['max'], chunk_dates.max())
        
        metadata = {
            'total_rows': total_rows,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'unique_comids': sorted(list(unique_comids)),
            'date_range': {
                'min': date_range['min'].isoformat() if date_range['min'] else None,
                'max': date_range['max'].isoformat() if date_range['max'] else None
            },
            'n_comids': len(unique_comids),
            'n_numeric_features': len(numeric_cols)
        }
        
        logging.info(f"数据分析完成:")
        logging.info(f"  - 总行数: {total_rows:,}")
        logging.info(f"  - COMID数量: {len(unique_comids):,}")
        logging.info(f"  - 数值特征: {len(numeric_cols)} 列")
        logging.info(f"  - 时间范围: {date_range['min']} 到 {date_range['max']}")
        
        return metadata
    
    def _convert_to_binary(self, metadata):
        """将CSV转换为高效的二进制格式"""
        logging.info("第二遍扫描：转换为二进制格式...")
        
        total_rows = metadata['total_rows']
        numeric_cols = metadata['numeric_cols']
        
        # 预分配NumPy数组
        numeric_data = np.empty((total_rows, len(numeric_cols)), dtype=np.float32)
        comid_array = np.empty(total_rows, dtype='<U20')  # 字符串数组
        date_array = np.empty(total_rows, dtype='datetime64[D]')
        
        # COMID索引：COMID -> [(start_idx, end_idx), ...]
        comid_index = {}
        
        current_row = 0
        chunk_iter = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        
        for chunk in tqdm(chunk_iter, desc="转换数据", total=total_rows//self.chunk_size + 1):
            chunk_size = len(chunk)
            
            # 处理数值数据
            numeric_data[current_row:current_row + chunk_size] = chunk[numeric_cols].values.astype(np.float32)
            
            # 处理COMID
            comids = chunk['COMID'].astype(str).values
            comid_array[current_row:current_row + chunk_size] = comids
            
            # 处理日期
            if 'date' in chunk.columns:
                dates = pd.to_datetime(chunk['date']).values.astype('datetime64[D]')
                date_array[current_row:current_row + chunk_size] = dates
            
            # 更新COMID索引
            for comid in np.unique(comids):
                comid_positions = np.where(comids == comid)[0] + current_row
                if comid not in comid_index:
                    comid_index[comid] = []
                comid_index[comid].extend(comid_positions.tolist())
            
            current_row += chunk_size
        
        # 优化COMID索引：转换为连续区间
        logging.info("优化COMID索引...")
        optimized_index = {}
        for comid, positions in tqdm(comid_index.items(), desc="优化索引"):
            positions = sorted(positions)
            ranges = []
            if len(positions) == 0:
                continue
                
            start = positions[0]
            end = positions[0]
            
            for pos in positions[1:]:
                if pos == end + 1:
                    end = pos
                else:
                    ranges.append((start, end + 1))  # end+1 for Python slicing
                    start = end = pos
            ranges.append((start, end + 1))
            optimized_index[comid] = ranges
        
        return {
            'numeric_data': numeric_data,
            'comid_array': comid_array,
            'date_array': date_array,
            'columns': numeric_cols
        }, optimized_index
    
    def _save_binary_data(self, data_arrays, comid_index, metadata):
        """保存二进制数据和索引"""
        logging.info("保存二进制文件...")
        
        # 保存主数据文件（关键：使用内存映射）
        np.save(self.output_dir / 'numeric_data.npy', data_arrays['numeric_data'])
        np.save(self.output_dir / 'comid_array.npy', data_arrays['comid_array'])
        np.save(self.output_dir / 'date_array.npy', data_arrays['date_array'])
        
        # 保存COMID索引
        with open(self.output_dir / 'comid_index.pkl', 'wb') as f:
            pickle.dump(comid_index, f)
        
        # 保存元数据
        metadata_copy = metadata.copy()
        metadata_copy['columns'] = data_arrays['columns']  # 添加列信息
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata_copy, f, indent=2, default=str)
        
        # 保存列名（独立文件，方便加载）
        with open(self.output_dir / 'columns.json', 'w') as f:
            json.dump(data_arrays['columns'], f)
        
        # 计算和输出统计信息
        data_size_mb = data_arrays['numeric_data'].nbytes / (1024**2)
        total_size_mb = sum([
            data_arrays['numeric_data'].nbytes,
            data_arrays['comid_array'].nbytes,
            data_arrays['date_array'].nbytes
        ]) / (1024**2)
        
        # 保存转换统计
        stats = {
            'conversion_completed': True,
            'numeric_data_size_mb': data_size_mb,
            'total_binary_size_mb': total_size_mb,
            'comid_count': len(comid_index),
            'compression_ratio': None,  # 可以计算与原CSV的压缩比
            'file_format': 'numpy_binary'
        }
        
        with open(self.output_dir / 'conversion_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"✓ 数据转换完成!")
        logging.info(f"✓ 数值数据大小: {data_size_mb:.1f} MB")
        logging.info(f"✓ 总文件大小: {total_size_mb:.1f} MB")
        logging.info(f"✓ COMID索引: {len(comid_index):,} 个")
        logging.info(f"✓ 输出目录: {self.output_dir}")


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('csv_conversion.log')
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSV到二进制格式转换器 - 为深度学习优化')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--chunk-size', type=int, default=100000, help='处理块大小 (默认: 100000)')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 验证输入文件
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    
    file_size_gb = os.path.getsize(args.input) / (1024**3)
    logging.info(f"输入文件大小: {file_size_gb:.1f} GB")
    
    if file_size_gb > 50:
        logging.warning(f"输入文件很大 ({file_size_gb:.1f} GB)，转换可能需要较长时间")
    
    # 创建转换器
    converter = CSVToBinaryConverter(args.input, args.output, args.chunk_size)
    
    try:
        # 执行转换
        output_dir = converter.convert()
        
        logging.info("=" * 60)
        logging.info("✓ 转换成功完成！")
        logging.info("=" * 60)
        logging.info(f"输出目录: {output_dir}")
        logging.info("")
        logging.info("接下来可以使用高效数据加载器:")
        logging.info("from efficient_data_loader import EfficientDataLoader")
        logging.info(f"loader = EfficientDataLoader('{output_dir}')")
        
    except Exception as e:
        logging.error(f"转换失败: {e}")
        logging.exception("详细错误信息:")
        sys.exit(1)


if __name__ == '__main__':
    main()
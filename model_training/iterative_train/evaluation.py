"""
evaluation.py - 高效评估和收敛检查模块

完全基于NumPy数组的评估系统，避免大DataFrame操作。
专为高效流量计算系统优化。
"""
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from ..gpu_memory_utils import TimingAndMemoryContext


class ConvergenceChecker:
    """收敛性检查器 - 基于NumPy数组，无DataFrame依赖"""
    
    def __init__(self, epsilon: float = 0.01, stability_threshold: float = 0.01, 
                history_window: int = 3):
        """
        初始化收敛性检查器
        
        参数:
            epsilon: 收敛阈值，当误差小于此值时认为收敛
            stability_threshold: 稳定性阈值，当误差变化率小于此值时认为趋势稳定
            history_window: 检查误差趋势稳定性时考虑的历史窗口大小
        """
        self.epsilon = epsilon
        self.stability_threshold = stability_threshold
        self.history_window = history_window
        self.error_history = []
        
    def check_convergence(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         iteration: int) -> Tuple[bool, Dict[str, float]]:
        """
        检查当前迭代是否达到收敛条件（纯NumPy实现）
        
        参数:
            y_true: 真实值数组
            y_pred: 预测值数组
            iteration: 当前迭代次数
            
        返回:
            (converged, stats): 是否收敛和统计信息
        """
        # 检查输入有效性
        if len(y_true) == 0 or len(y_pred) == 0:
            logging.warning("警告：输入数组为空，无法评估收敛性")
            return False, None
            
        if len(y_true) != len(y_pred):
            logging.warning("警告：真实值和预测值数组长度不匹配")
            return False, None
        
        # 过滤有效数据（避免NaN和Inf）
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            logging.warning("警告：没有有效的观测数据，无法评估收敛性")
            return False, None
        
        # 提取有效数据
        valid_y_true = y_true[valid_mask]
        valid_y_pred = y_pred[valid_mask]
        
        # 计算残差
        residual = valid_y_true - valid_y_pred
        
        # 计算误差统计量（高效NumPy实现）
        mae = np.mean(np.abs(residual))
        mse = np.mean(np.square(residual))
        rmse = np.sqrt(mse)
        max_resid = np.max(np.abs(residual))
        
        # 汇总统计信息
        stats = {
            'iteration': iteration,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'max_resid': float(max_resid),
            'valid_data_points': int(valid_count)
        }
        
        # 记录误差历史
        self.error_history.append(stats)
        
        # 输出误差信息
        logging.info(f"迭代 {iteration+1} 误差统计 (基于 {valid_count} 个有效观测点):")
        logging.info(f"  平均绝对误差 (MAE): {mae:.4f}")
        logging.info(f"  均方误差 (MSE): {mse:.4f}")
        logging.info(f"  均方根误差 (RMSE): {rmse:.4f}")
        logging.info(f"  最大绝对残差: {max_resid:.4f}")
        
        # 检查收敛条件
        if mae < self.epsilon:
            logging.info(f"收敛! 平均绝对误差 ({mae:.4f}) 小于阈值 ({self.epsilon})")
            return True, stats
        
        # 检查误差趋势是否稳定
        if self.check_error_trend_stability():
            return True, stats
            
        return False, stats
    
    def check_error_trend_stability(self) -> bool:
        """检查误差趋势是否稳定"""
        if len(self.error_history) < self.history_window:
            return False
            
        # 获取最近几轮的误差
        recent_errors = [entry['mae'] for entry in self.error_history[-self.history_window:]]
        
        # 计算误差变化率
        error_changes = []
        for i in range(1, len(recent_errors)):
            prev_error = recent_errors[i-1]
            if prev_error > 0:
                change = (prev_error - recent_errors[i]) / prev_error
                error_changes.append(change)
        
        # 检查变化率是否都小于阈值
        if error_changes and all(abs(change) < self.stability_threshold for change in error_changes):
            logging.info(f"收敛! 误差变化趋于稳定，最近几轮MAE: {recent_errors}")
            return True
        
        return False
    
    def get_error_history(self) -> List[Dict[str, float]]:
        """获取完整的误差历史记录"""
        return self.error_history


class EfficientDataValidator:
    """
    高效数据验证器 - 基于二进制数据和NumPy数组
    
    替代传统的DataFrame异常检查，直接在二进制结果上进行验证
    """
    
    def __init__(self, max_abnormal_value: float = 1e6, 
                max_allowed_percent: float = 1.0):
        """
        初始化数据验证器
        
        参数:
            max_abnormal_value: 允许的最大异常值绝对值
            max_allowed_percent: 允许的最大异常比例（百分比）
        """
        self.max_abnormal_value = max_abnormal_value
        self.max_allowed_percent = max_allowed_percent
    
    def validate_binary_results(self, 
                               result_dir: str,
                               target_col: str,
                               iteration: int) -> Tuple[bool, Dict]:
        """
        验证二进制计算结果（无DataFrame）
        
        参数:
            result_dir: 二进制结果目录
            target_col: 目标列名
            iteration: 迭代次数
            
        返回:
            (is_valid, report): 数据是否有效和异常报告
        """
        logging.info(f"验证迭代 {iteration} 的二进制结果质量...")
        
        report = {
            "iteration": iteration,
            "target_col": target_col,
            "is_valid": True,
            "anomaly_stats": {},
            "recommendations": []
        }
        
        try:
            # 加载二进制结果（内存映射模式）
            y_up_path = os.path.join(result_dir, f'y_up_{target_col}.npy')
            y_n_path = os.path.join(result_dir, f'y_n_{target_col}.npy')
            
            if not (os.path.exists(y_up_path) and os.path.exists(y_n_path)):
                report["is_valid"] = False
                report["recommendations"].append("二进制结果文件缺失")
                return False, report
            
            # 内存映射加载（避免全量载入内存）
            y_up_data = np.load(y_up_path, mmap_mode='r')
            y_n_data = np.load(y_n_path, mmap_mode='r')
            
            # 验证数组 - 使用高效的NumPy操作
            for array_name, array_data in [('y_up', y_up_data), ('y_n', y_n_data)]:
                stats = self._validate_numpy_array(array_data, array_name)
                report["anomaly_stats"][array_name] = stats
                
                # 检查异常比例
                total_elements = array_data.size
                total_anomalies = stats['nan_count'] + stats['inf_count'] + stats['extreme_count']
                anomaly_percent = (total_anomalies / total_elements * 100) if total_elements > 0 else 0
                
                if anomaly_percent > self.max_allowed_percent:
                    report["is_valid"] = False
                    report["recommendations"].append(f"{array_name}异常值过多: {anomaly_percent:.2f}%")
                    logging.error(f"数组 {array_name} 异常值过多: {anomaly_percent:.2f}%")
                elif anomaly_percent > 0.01:  # 轻微异常也要记录
                    logging.warning(f"数组 {array_name} 包含异常值: {anomaly_percent:.2f}%")
            
            if report["is_valid"]:
                logging.info(f"迭代 {iteration} 二进制结果验证通过")
            else:
                logging.error(f"迭代 {iteration} 二进制结果验证失败")
                
        except Exception as e:
            report["is_valid"] = False
            report["recommendations"].append(f"验证过程出错: {e}")
            logging.error(f"二进制结果验证出错: {e}")
        
        return report["is_valid"], report
    
    def _validate_numpy_array(self, array: np.ndarray, name: str) -> Dict:
        """
        验证单个NumPy数组的数据质量
        
        参数:
            array: 要验证的NumPy数组
            name: 数组名称
            
        返回:
            验证统计信息字典
        """
        # 高效计算异常值统计
        nan_count = int(np.sum(np.isnan(array)))
        inf_count = int(np.sum(np.isinf(array)))
        
        # 计算极端值（排除NaN和Inf）
        finite_mask = np.isfinite(array)
        finite_data = array[finite_mask]
        extreme_count = int(np.sum(np.abs(finite_data) > self.max_abnormal_value)) if finite_data.size > 0 else 0
        
        # 计算统计量
        stats = {
            'name': name,
            'total_elements': int(array.size),
            'nan_count': nan_count,
            'inf_count': inf_count,
            'extreme_count': extreme_count,
            'min_value': float(np.min(finite_data)) if finite_data.size > 0 else 0.0,
            'max_value': float(np.max(finite_data)) if finite_data.size > 0 else 0.0,
            'mean_value': float(np.mean(finite_data)) if finite_data.size > 0 else 0.0,
        }
        
        return stats
    
    def fix_binary_results_if_needed(self, 
                                   result_dir: str,
                                   target_col: str,
                                   reasonable_max: float = 100.0) -> str:
        """
        如果需要，修复二进制结果中的异常值
        
        参数:
            result_dir: 二进制结果目录
            target_col: 目标列名
            reasonable_max: 合理的最大值
            
        返回:
            修复后的结果目录路径
        """
        logging.info("检查并修复二进制结果中的异常值...")
        
        # 加载数据
        y_up_path = os.path.join(result_dir, f'y_up_{target_col}.npy')
        y_n_path = os.path.join(result_dir, f'y_n_{target_col}.npy')
        
        if not (os.path.exists(y_up_path) and os.path.exists(y_n_path)):
            logging.error("二进制结果文件不存在，无法修复")
            return result_dir
        
        try:
            # 加载数据（可写模式）
            y_up_data = np.load(y_up_path)
            y_n_data = np.load(y_n_path)
            
            # 修复异常值（就地操作）
            y_up_data = np.nan_to_num(y_up_data, nan=0.0, posinf=reasonable_max, neginf=-reasonable_max)
            y_n_data = np.nan_to_num(y_n_data, nan=0.0, posinf=reasonable_max, neginf=-reasonable_max)
            
            # 限制极端值
            np.clip(y_up_data, -reasonable_max, reasonable_max, out=y_up_data)
            np.clip(y_n_data, -reasonable_max, reasonable_max, out=y_n_data)
            
            # 创建修复后的目录
            fixed_result_dir = result_dir + "_fixed"
            os.makedirs(fixed_result_dir, exist_ok=True)
            
            # 保存修复后的数据
            np.save(os.path.join(fixed_result_dir, f'y_up_{target_col}.npy'), y_up_data)
            np.save(os.path.join(fixed_result_dir, f'y_n_{target_col}.npy'), y_n_data)
            
            # 复制元数据
            import shutil
            metadata_src = os.path.join(result_dir, 'metadata.json')
            metadata_dst = os.path.join(fixed_result_dir, 'metadata.json')
            if os.path.exists(metadata_src):
                shutil.copy2(metadata_src, metadata_dst)
            
            logging.info(f"异常值修复完成，修复后结果保存至: {fixed_result_dir}")
            return fixed_result_dir
            
        except Exception as e:
            logging.error(f"修复二进制结果时出错: {e}")
            return result_dir


class MinimalVisualizer:
    """
    最小化可视化器 - 仅生成必要的验证图表
    
    专注于核心验证功能，避免复杂的DataFrame操作
    """
    
    @staticmethod
    def create_convergence_plot(convergence_checker: ConvergenceChecker, 
                              output_dir: str, 
                              model_version: str):
        """
        创建收敛趋势图
        
        参数:
            convergence_checker: 收敛检查器实例
            output_dir: 输出目录
            model_version: 模型版本
        """
        error_history = convergence_checker.get_error_history()
        
        if len(error_history) < 2:
            logging.info("误差历史数据不足，跳过收敛图生成")
            return
        
        # 提取数据
        iterations = [entry['iteration'] for entry in error_history]
        maes = [entry['mae'] for entry in error_history]
        rmses = [entry['rmse'] for entry in error_history]
        
        # 创建收敛图
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, maes, 'b-o', label='MAE')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE Convergence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, rmses, 'r-s', label='RMSE')
        plt.xlabel('Iteration')
        plt.ylabel('Root Mean Square Error')
        plt.title('RMSE Convergence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle(f'Training Convergence - {model_version}', fontsize=14)
        plt.tight_layout()
        
        # 保存图表
        convergence_plot_path = os.path.join(output_dir, f'convergence_plot_{model_version}.png')
        plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"收敛图已保存至: {convergence_plot_path}")
    
    @staticmethod
    def create_simple_validation_plot(y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    iteration: int,
                                    target_col: str,
                                    output_dir: str,
                                    model_version: str):
        """
        创建简单的验证散点图
        
        参数:
            y_true: 真实值数组
            y_pred: 预测值数组
            iteration: 迭代次数
            target_col: 目标列名
            output_dir: 输出目录
            model_version: 模型版本
        """
        # 过滤有效数据
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        
        if np.sum(valid_mask) < 10:
            logging.warning("有效数据点太少，跳过验证图生成")
            return
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # 创建散点图
        plt.figure(figsize=(8, 8))
        
        # 采样绘制（如果数据点太多）
        if len(y_true_valid) > 5000:
            sample_indices = np.random.choice(len(y_true_valid), 5000, replace=False)
            y_true_sample = y_true_valid[sample_indices]
            y_pred_sample = y_pred_valid[sample_indices]
        else:
            y_true_sample = y_true_valid
            y_pred_sample = y_pred_valid
        
        plt.scatter(y_true_sample, y_pred_sample, alpha=0.6, s=10)
        
        # 添加1:1线
        min_val = min(np.min(y_true_sample), np.min(y_pred_sample))
        max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
        
        # 计算性能指标
        mse = np.mean((y_true_valid - y_pred_valid)**2)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
        r2 = correlation**2
        
        # 添加性能指标文本
        plt.text(0.05, 0.95, 
                f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\nN: {len(y_true_valid)}', 
                transform=plt.gca().transAxes, 
                fontsize=12, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel(f'Observed {target_col}')
        plt.ylabel(f'Predicted {target_col}')
        plt.title(f'{target_col} Validation - Iteration {iteration} ({model_version})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存图表
        plot_path = os.path.join(output_dir, f'validation_iter{iteration}_{target_col}_{model_version}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"验证图已保存至: {plot_path}")


def get_evaluation_memory_usage() -> Dict[str, float]:
    """获取评估模块的内存使用情况"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'evaluation_rss_mb': memory_info.rss / 1024 / 1024,
        'evaluation_mode': 'efficient_numpy_only',
        'dataframe_usage': 'eliminated'
    }
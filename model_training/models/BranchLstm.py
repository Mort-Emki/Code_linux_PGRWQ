"""
branch_lstm.py - 分支LSTM模型实现

该模块继承CatchmentModel父类，提供多分支LSTM模型的完整实现。
保留了父类定义的接口，并实现了所有必要的抽象方法。
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from PGRWQI.model_training.gpu_memory_utils import log_memory_usage, TimingAndMemoryContext 
from PGRWQI.model_training.models.models import CatchmentModel

# =============================================================================
# 网络模型定义
# =============================================================================

class MultiBranchLSTM(nn.Module):
    """
    多分支LSTM模型
    
    结合时间序列数据和静态属性数据，通过LSTM和MLP分支进行联合建模
    """
    def __init__(self, input_dim, hidden_size, num_layers, attr_dim, fc_dim, output_dim=1, use_attr=True):
        """
        初始化多分支LSTM模型
        
        参数:
            input_dim: 时间序列特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            attr_dim: 属性数据维度
            fc_dim: 属性数据全连接层输出维度
            output_dim: 模型输出维度（默认1）
            use_attr: 是否使用属性数据
        """
        super().__init__()
        self.use_attr = use_attr

        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )

        if self.use_attr:
            # 属性分支：MLP
            self.attr_fc = nn.Sequential(
                nn.Linear(attr_dim, fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU()
            )
            self.final_fc = nn.Linear(hidden_size + fc_dim, output_dim)
        else:
            self.final_fc = nn.Linear(hidden_size, output_dim)
        
        # 打印模型架构和参数数量
        print(f"模型架构初始化:")
        print(f" - LSTM: input_dim={input_dim}, hidden_size={hidden_size}, num_layers={num_layers}")
        if self.use_attr:
            print(f" - 属性网络: attr_dim={attr_dim}, fc_dim={fc_dim}")
        print(f" - 输出维度: {output_dim}")
        
        # 计算并打印参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数量: {total_params:,}")

    def forward(self, x_ts, x_attr):
        """
        前向传播
        
        参数:
            x_ts: 时间序列输入, 形状为 (batch_size, seq_len, input_dim)
            x_attr: 属性输入, 形状为 (batch_size, attr_dim)
            
        返回:
            模型输出
        """
        lstm_out, _ = self.lstm(x_ts)
        ts_feat = lstm_out[:, -1, :]  # 取时间序列最后一时刻特征

        if self.use_attr:
            attr_feat = self.attr_fc(x_attr)
            combined = torch.cat([ts_feat, attr_feat], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(ts_feat)

        return out.squeeze(-1)

# =============================================================================
# 分支LSTM模型 - 继承自CatchmentModel
# =============================================================================

class BranchLSTMModel(CatchmentModel):
    """
    分支LSTM模型实现 - 继承自CatchmentModel基类
    
    实现父类定义的接口，并提供LSTM特有的功能
    """
    def __init__(self, input_dim, hidden_size=64, num_layers=1, attr_dim=20, 
                 fc_dim=32, device='cpu', memory_check_interval=5):
        """
        初始化分支LSTM模型
        
        参数:
            input_dim: 时间序列输入维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            attr_dim: 属性数据维度
            fc_dim: 全连接层维度
            device: 训练设备
            memory_check_interval: 内存检查间隔(epochs)
        """
        # 调用父类构造函数
        super(BranchLSTMModel, self).__init__(
            model_type='lstm',
            device=device,
            memory_check_interval=memory_check_interval
        )
        
        # 存储模型参数
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attr_dim = attr_dim
        self.fc_dim = fc_dim
        
        # 记录初始内存状态
        if self.device == 'cuda':
            log_memory_usage("[BranchLSTM模型初始化] ")
            
        # 初始化LSTM模型
        self._init_model()
    
    def _init_model(self):
        """初始化LSTM模型"""
        # 创建多分支LSTM模型
        self.base_model = MultiBranchLSTM(
            input_dim=self.input_dim, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            attr_dim=self.attr_dim, 
            fc_dim=self.fc_dim
        ).to(self.device)
        
        # 在GPU上测试模型
        if self.device == 'cuda':
            self._test_model_on_device()
    
    def _test_model_on_device(self):
        """测试模型是否正确加载到设备上"""
        dummy_ts = torch.zeros((1, 10, self.input_dim), device=self.device)
        dummy_attr = torch.zeros((1, self.attr_dim), device=self.device)
        with torch.no_grad():
            _ = self.base_model(dummy_ts, dummy_attr)
        print(f"已在{self.device}上测试模型, 使用虚拟输入")
        
        # 打印每个参数的设备以确认
        print("参数设备:")
        for name, param in self.base_model.named_parameters():
            print(f" - {name}: {param.device}")
        
    def train_model(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
                comid_arr_val=None, X_ts_val=None, Y_val=None, 
                epochs=10, lr=1e-3, patience=3, batch_size=32, early_stopping=False):
        """
        训练分支LSTM模型 - 仅支持流式训练
        
        参数:
            attr_dict: 属性字典
            X_ts_train: 流式训练数据迭代器
            Y_train: 未使用（流式模式）
            X_ts_val: 流式验证数据迭代器
            Y_val: 未使用（流式模式）
            epochs: 训练轮数
            lr: 学习率
            patience: 早停耐心值
            batch_size: 未使用（由迭代器控制）
            early_stopping: 是否启用早停机制
        """
        
        # 仅支持流式训练
        if not hasattr(X_ts_train, '__iter__'):
            raise ValueError("仅支持流式训练迭代器，请使用DataHandler.prepare_streaming_training_data()创建迭代器")
        
        validation_iterator = X_ts_val if X_ts_val is not None else None
        return self.train_model_streaming(
            streaming_iterator=X_ts_train,
            validation_iterator=validation_iterator,
            epochs=epochs,
            lr=lr,
            patience=patience,
            early_stopping=early_stopping
        )
    
    
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
                for batch_idx, batch_data in enumerate(streaming_iterator):
                    try:
                        # 流式训练迭代器返回: (X_ts, Y, COMIDs, batch_comids)
                        X_ts_batch, Y_batch, COMIDs_batch, batch_comids = batch_data
                        
                        # 构建属性数据（从属性字典获取）
                        X_attr_batch = self._get_attr_data_for_batch(COMIDs_batch)
                        
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
    
    def _get_attr_data_for_batch(self, comids_batch):
        """为给定的COMID批次获取属性数据"""
        # 创建属性数据数组
        if hasattr(self, '_cached_attr_dict') and self._cached_attr_dict:
            # 如果有缓存的属性字典，使用它
            attr_dict = self._cached_attr_dict
            attr_dim = next(iter(attr_dict.values())).shape[0]
        else:
            # 默认属性维度
            attr_dim = 10
            attr_dict = {}
        
        X_attr_batch = np.zeros((len(comids_batch), attr_dim), dtype=np.float32)
        
        for i, comid in enumerate(comids_batch):
            comid_str = str(comid)
            if comid_str in attr_dict:
                X_attr_batch[i] = attr_dict[comid_str]
        
        return X_attr_batch
    
    def set_attr_dict(self, attr_dict):
        """设置属性字典（由模型管理器调用）"""
        self._cached_attr_dict = attr_dict
    
    def _validate_streaming(self, validation_iterator, criterion):
        """流式验证"""
        import torch
        import gc
        
        self.base_model.eval()
        total_val_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in validation_iterator:
                try:
                    # 流式验证迭代器返回: (X_ts, Y, COMIDs, batch_comids)
                    X_ts_batch, Y_batch, COMIDs_batch, batch_comids = batch_data
                    
                    # 构建属性数据
                    X_attr_batch = self._get_attr_data_for_batch(COMIDs_batch)
                    
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


    def predict(self, X_ts, X_attr):
        """
        批量预测
        
        实现父类的抽象方法，执行LSTM模型的预测流程
        
        参数:
            X_ts: 时间序列输入, 形状为(N, T, D)
            X_attr: 属性输入, 形状为(N, attr_dim)
            
        返回:
            预测结果, 形状为(N,)
        """
        with TimingAndMemoryContext("批量预测"):
            self.base_model.eval()
            
            # 确保模型在正确的设备上
            if self.device == 'cuda':
                self.base_model = self.base_model.to(self.device)
            
            total_samples = X_ts.shape[0]
            
            # 计算合适的批处理大小
            batch_size = self._calculate_safe_batch_size(X_ts, X_attr)
            
            if self.device == 'cuda':
                log_memory_usage(f"[预测开始] 处理 {total_samples} 个样本")
            
            # 分批进行预测
            all_preds = []
            current_batch_size = batch_size
            
            i = 0
            while i < total_samples:
                try:
                    # 尝试使用当前批处理大小
                    end_idx = min(i + current_batch_size, total_samples)
                    
                    # 在正确的设备上创建张量
                    X_ts_torch = torch.tensor(X_ts[i:end_idx], dtype=torch.float32, device=self.device)
                    X_attr_torch = torch.tensor(X_attr[i:end_idx], dtype=torch.float32, device=self.device)
                    
                    # 获取预测
                    with torch.no_grad():
                        batch_preds = self.base_model(X_ts_torch, X_attr_torch)
                    
                    # 存储预测结果
                    all_preds.append(batch_preds.cpu().numpy())
                    
                    # 释放内存
                    del X_ts_torch
                    del X_attr_torch
                    torch.cuda.empty_cache()
                    
                    # 移动到下一批
                    i = end_idx
                    
                    # 定期记录进度
                    # if i % (10 * current_batch_size) == 0 or i == total_samples:
                    #     # print(f"已处理 {i}/{total_samples} 个样本 ({i/total_samples*100:.1f}%)")
                    
                except RuntimeError as e:
                    # 检查是否是内存不足错误
                    if "CUDA out of memory" in str(e):
                        # 减少批处理大小并重试
                        torch.cuda.empty_cache()
                        old_batch_size = current_batch_size
                        current_batch_size = max(10, current_batch_size // 2)
                        print(f"⚠️ 批处理大小 {old_batch_size} 内存不足。减小到 {current_batch_size}")
                        
                        # 如果批处理大小已经很小，可能存在其他问题
                        if current_batch_size < 100:
                            print("⚠️ 警告: 需要非常小的批处理大小。如果情况继续，请考虑使用CPU")
                    else:
                        # 不是内存错误，重新抛出
                        raise
            
            if len(all_preds) == 0:
                raise RuntimeError("未能成功处理任何批次")
                
            return np.concatenate(all_preds)
    
    def predict_simple(self, X_ts, X_attr):
        """
        简化版本的预测函数（无批量处理）

        参数:
            X_ts: 时间序列输入, 形状为(N, T, D)
            X_attr: 属性输入, 形状为(N, attr_dim)

        返回:
            预测结果, 形状为(N,)
        """
        self.base_model.eval()

        # 将数据移动到模型所在设备
        X_ts_torch = torch.tensor(X_ts, dtype=torch.float32, device=self.device)
        X_attr_torch = torch.tensor(X_attr, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = self.base_model(X_ts_torch, X_attr_torch)

        return preds.cpu().numpy()

    def save_model(self, path):
        """
        保存模型
        
        实现父类的抽象方法，保存LSTM模型
        
        参数:
            path: 保存路径
        """
        torch.save(self.base_model.state_dict(), path)
        print(f"模型已保存到 {path}")
        
        # 保存后记录内存
        if self.device == 'cuda':
            log_memory_usage("[模型已保存] ")

    def load_model(self, path):
        """
        加载模型
        
        实现父类的抽象方法，加载LSTM模型
        
        参数:
            path: 模型路径
        """
        with TimingAndMemoryContext("加载模型"):
            self.base_model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"模型已从 {path} 加载")
            
            # 加载后记录内存
            if self.device == 'cuda':
                log_memory_usage("[模型已加载] ")
    
    def get_model_info(self):
        """
        获取模型信息
        
        扩展父类方法，添加LSTM特有信息
        
        返回:
            包含模型信息的字典
        """
        info = super().get_model_info()
        info.update({
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "attr_dim": self.attr_dim,
            "fc_dim": self.fc_dim
        })
        return info

# =============================================================================
# 创建模型实例的工厂函数
# =============================================================================

def create_branch_lstm_model(input_dim, hidden_size=64, num_layers=1, attr_dim=20, 
                             fc_dim=32, device='cuda', memory_check_interval=5):
    """
    创建分支LSTM模型的工厂函数
    
    参数:
        input_dim: 输入维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        attr_dim: 属性维度
        fc_dim: 全连接层维度
        device: 训练设备('cpu'或'cuda')
        memory_check_interval: 内存检查间隔(epochs)
        
    返回:
        BranchLSTMModel实例
    """
    return BranchLSTMModel(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attr_dim=attr_dim,
        fc_dim=fc_dim,
        device=device,
        memory_check_interval=memory_check_interval
    )
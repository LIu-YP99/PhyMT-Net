import os

class Config:
    # =========================================================================
    # 1. 路径配置 (Path Configuration)
    # =========================================================================
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 【训练数据路径】
    # 指向你生成的合成数据 (包含 waveforms.json 的文件夹)
    # 如果你要切换到真实数据进行测试，修改这里即可
    DATA_ROOT = os.path.join(PROJECT_ROOT, "../data", "output_synthetic")
    
    # 【Instaseis 数据库路径】
    # 用于 vis_waveform_fit.py 进行正演验证
    # 根据你提供的路径 "../generate_wave"，我将其映射如下：
    INSTASEIS_DB_PATH = os.path.join(PROJECT_ROOT, "../", "generate_wave")
    
    # JSON 文件映射
    JSON_PATHS = {
        "waveforms": os.path.join(DATA_ROOT, "waveforms.json"),
        "source":    os.path.join(DATA_ROOT, "sources.json"),
        "grid":      os.path.join(DATA_ROOT, "grid_points.json"),
        "receivers": os.path.join(DATA_ROOT, "receivers.json") 
    }
    
    # 检查点与日志
    SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    
    # =========================================================================
    # 2. 数据核心参数 (Data Spec - 必须与生成脚本一致)
    # =========================================================================
    # 动态台阵最大支持数
    MAX_STATIONS = 24       
    
    # 采样率 (Hz)
    SAMPLE_RATE = 2.0       
    
    # 【关键补全】采样点数
    # 很多脚本用 N_POINTS，有的用 SEQ_LEN，这里统一
    N_POINTS = 512
    SEQ_LEN = 512           
    
    # 【关键补全】时间长度 (秒)
    DURATION = N_POINTS / SAMPLE_RATE  # 256.0 秒
    
    # 【关键补全】频率范围 (Hz)
    # Sim-to-Real 的甜点区
    FREQ_MIN = 0.02
    FREQ_MAX = 0.25         
    
    # 几何特征维度 [sin, cos, dist, depth]
    GEO_DIM = 4             
    MAX_DIST_KM = 800.0     
    CHANNELS = 3            

    # =========================================================================
    # 3. 训练超参数 (Training Hyperparameters)
    # =========================================================================
    BATCH_SIZE = 128
    EPOCHS = 300
    NUM_WORKERS = 16
    
    # 学习率: 建议用 1e-4 配合 OneCycleLR，1e-5 可能收敛太慢
    LEARNING_RATE = 5e-5    
    
    # 权重衰减: 防止过拟合
    WEIGHT_DECAY = 1e-4     
    
    # Loss 权重
    # 建议 Alpha=10.0 以强迫模型学习形状，Beta=0.1 或 1.0 防止震级主导
    LOSS_ALPHA = 2      # Shape 权重
    LOSS_BETA = 1        # Magnitude 权重
    
    # =========================================================================
    # 4. 增强与鲁棒性 (Augmentation)
    # =========================================================================
    # 台站丢弃概率 (模拟真实坏台)
    STATION_DROPOUT_PROB = 0.1 
    
    # 极性翻转
    # 如果你的数据集是"物理平衡"的(包含塌缩)，建议设为 0.5
    # 如果只包含爆炸，设为 0.0
    AUGMENT_PROB = 0.5      
    
    # 噪声注入
    NOISE_LEVEL = 0
    
    # =========================================================================
    # 5. 模型结构参数 (Model Architecture)
    # =========================================================================
    # 即使代码里写死，放在 Config 里也方便以后改
    CNN_FILTERS = [32, 64, 128]
    FEATURE_DIM = 256   
    TRANSFORMER_HEADS = 8
    TRANSFORMER_LAYERS = 3
    
    # =========================================================================
    # 6. 自动化控制 (Automation)
    # =========================================================================
    EARLY_STOPPING_PATIENCE = 300
    
    # 断点续训
    RESUME_TRAINING = False
    RESUME_CHECKPOINT = "best_model_20251202_2257.pth"
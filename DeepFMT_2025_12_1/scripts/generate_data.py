import os
import json
import numpy as np
import obspy
import instaseis
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ================= 配置区域 (保持 0.25Hz 最优配置) =================
class GenConfig:
    # 1. 路径
    DB_PATH = "../generate_wave"  # 修改为你实际的数据库路径
    OUTPUT_DIR = "./data/output_synthetic/train"
    WAVEFORM_DIR = os.path.join(OUTPUT_DIR, "waveforms")
    
    # 2. 模拟参数
    FREQ_MIN = 0.02
    FREQ_MAX = 0.25
    SAMPLE_RATE = 2.0
    N_POINTS = 512
    DURATION = N_POINTS / SAMPLE_RATE
    
    # 3. 空间网格
    LAT_RANGE = (33.0, 34.0)
    LON_RANGE = (-116.8, -115.8)
    DEPTH_RANGE_KM = (2.0, 20.0)
    GRID_SHAPE = (12, 12, 9) 
    
    # 4. 数据量
    SOURCES_PER_GRID = 50 
    
    # 5. 动态台站
    N_STATIONS_PER_EVENT = 24 
    MIN_DIST_KM = 10.0
    MAX_DIST_KM = 200.0
    
    # 6. 并行设置
    # 默认使用 CPU 核心数 - 2，防止卡死系统
    MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 工具函数 =================

def get_mt_from_hudson(T, k, M0=1.0):
    iso = k * M0
    remaining_M0 = M0 * (1 - abs(k))
    if remaining_M0 < 1e-9:
        eig = np.array([iso/3]*3)
    else:
        l_dev = np.array([1.0 - 0.5*abs(T), T, -1.0 + 0.5*abs(T)])
        l_dev = l_dev / np.max(np.abs(l_dev)) * remaining_M0
        eig = l_dev + iso/3.0
    Q = R.random().as_matrix()
    M = Q @ np.diag(eig) @ Q.T
    return [M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]]

def ned_to_use(m_ned):
    mxx, myy, mzz, mxy, mxz, myz = m_ned
    return mzz, mxx, myy, mxz, -myz, -mxy

def generate_random_receivers(center_lat, center_lon, n):
    receivers = []
    meta = []
    for i in range(n):
        dist_km = np.random.uniform(GenConfig.MIN_DIST_KM, GenConfig.MAX_DIST_KM)
        azimuth = np.random.uniform(0, 360)
        delta_deg = dist_km / 111.19
        rec_lat = center_lat + delta_deg * np.cos(np.deg2rad(azimuth))
        rec_lon = center_lon + delta_deg * np.sin(np.deg2rad(azimuth)) / np.cos(np.deg2rad(center_lat))
        
        code = f"S{i:02d}"
        rec = instaseis.Receiver(latitude=rec_lat, longitude=rec_lon, network="SY", station=code)
        receivers.append(rec)
        meta.append({"code": f"SY.{code}", "lat": rec_lat, "lon": rec_lon})
    return receivers, meta

# ================= 多进程 Worker 逻辑 =================

# 全局变量，用于在每个子进程中只加载一次 DB
db = None

def init_worker(db_path):
    """每个子进程初始化时调用：加载数据库"""
    global db
    try:
        db = instaseis.open_db(db_path)
    except Exception as e:
        print(f"Worker Init Error: {e}")

def process_grid_point(gp):
    """
    处理单个网格点任务：生成该点下的所有震源和波形
    """
    global db
    if db is None: return None # DB 加载失败
    
    g_idx = gp["idx"]
    local_sources = {}
    local_waveforms = []
    
    # 固定发震时刻
    origin_time = obspy.UTCDateTime(2020, 1, 1, 0, 0, 0)
    
    # 随机生成该 Grid Point 的台站布局
    # 注意：为了让每个 Grid 的随机性独立，这里直接调用 numpy 即可
    # 多进程下每个进程的 Random State 是独立的
    current_receivers, current_rec_meta = generate_random_receivers(
        gp["lat"], gp["lon"], GenConfig.N_STATIONS_PER_EVENT
    )
    
    local_sources[str(g_idx)] = {}
    
    for m_idx in range(GenConfig.SOURCES_PER_GRID):
        # 1. 采样震源参数
        ratio = m_idx / GenConfig.SOURCES_PER_GRID
        if ratio < 0.33: T, k = 0, 0
        elif ratio < 0.66:
            k = np.random.uniform(0.5, 1.0) * (1 if np.random.random()>0.5 else -1)
            T = np.random.uniform(-0.5, 0.5)
        else:
            k = np.random.uniform(-1, 1)
            T = np.random.uniform(-1, 1)
        
        mw = np.random.uniform(3.5, 6.0)
        m0 = 10**(1.5 * mw + 9.1)
        mt_ned = get_mt_from_hudson(T, k, m0)
        
        # 记录 Source
        local_sources[str(g_idx)][str(m_idx)] = {
            "mechanism_idx": m_idx,
            "moment_tensor": mt_ned,
            "magnitude": mw,
            "hudson_k": k, "hudson_T": T
        }
        
        # 2. Instaseis Source
        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp = ned_to_use(mt_ned)
        source = instaseis.Source(
            latitude=gp["lat"], longitude=gp["lon"], depth_in_m=gp["depth_m"],
            m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
            origin_time=origin_time
        )
        
        # 3. 生成波形
        for i, receiver in enumerate(current_receivers):
            try:
                st = db.get_seismograms(source=source, receiver=receiver)
                
                # 处理
                st.filter('bandpass', freqmin=GenConfig.FREQ_MIN, freqmax=GenConfig.FREQ_MAX, corners=4, zerophase=True)
                st.resample(GenConfig.SAMPLE_RATE)
                st.trim(starttime=origin_time, endtime=origin_time + GenConfig.DURATION, pad=True, fill_value=0)
                
                # 对齐点数
                for tr in st:
                    if tr.stats.npts < GenConfig.N_POINTS:
                        tr.data = np.pad(tr.data, (0, GenConfig.N_POINTS - tr.stats.npts), 'constant')
                    elif tr.stats.npts > GenConfig.N_POINTS:
                        tr.data = tr.data[:GenConfig.N_POINTS]
                
                # 保存
                filename = f"grid{g_idx:04d}_mech{m_idx:02d}_sta{i:02d}.mseed"
                save_path = os.path.join(GenConfig.WAVEFORM_DIR, filename)
                st.write(save_path, format="MSEED")
                
                # 记录 Meta
                local_waveforms.append({
                    "grid_point_idx": g_idx,
                    "mechanism_idx": m_idx,
                    "receiver_code": current_rec_meta[i]['code'],
                    "receiver_lat": current_rec_meta[i]['lat'],
                    "receiver_lon": current_rec_meta[i]['lon'],
                    "waveform_file": os.path.join("train", "waveforms", filename),
                    "split": "train"
                })
            except Exception:
                pass
                
    return local_sources, local_waveforms

# ================= 主程序 =================

def main():
    if not os.path.exists(GenConfig.WAVEFORM_DIR):
        os.makedirs(GenConfig.WAVEFORM_DIR)
        
    print(f"🚀 Starting Multi-process Generation with {GenConfig.MAX_WORKERS} workers.")
    print(f"Target: {GenConfig.FREQ_MAX}Hz, {GenConfig.SAMPLE_RATE}Hz Sampling")

    # 1. 生成网格列表
    lats = np.linspace(GenConfig.LAT_RANGE[0], GenConfig.LAT_RANGE[1], GenConfig.GRID_SHAPE[0])
    lons = np.linspace(GenConfig.LON_RANGE[0], GenConfig.LON_RANGE[1], GenConfig.GRID_SHAPE[1])
    deps = np.linspace(GenConfig.DEPTH_RANGE_KM[0], GenConfig.DEPTH_RANGE_KM[1], GenConfig.GRID_SHAPE[2]) * 1000 
    
    grid_points = []
    grid_idx = 0
    for lat in lats:
        for lon in lons:
            for dep in deps:
                grid_points.append({"idx": grid_idx, "lat": lat, "lon": lon, "depth_m": dep})
                grid_idx += 1
    
    print(f"Total Grid Points: {len(grid_points)}")
    print(f"Total Waveforms to generate: {len(grid_points) * GenConfig.SOURCES_PER_GRID * GenConfig.N_STATIONS_PER_EVENT}")

    # 2. 并行处理
    all_sources = {}
    all_waveforms = []
    
    # 使用 ProcessPoolExecutor
    # initializer=init_worker 确保每个进程只打开一次 DB
    with ProcessPoolExecutor(max_workers=GenConfig.MAX_WORKERS, 
                             initializer=init_worker, 
                             initargs=(GenConfig.DB_PATH,)) as executor:
        
        # 提交任务
        futures = {executor.submit(process_grid_point, gp): gp for gp in grid_points}
        
        # 进度条
        for future in tqdm(as_completed(futures), total=len(grid_points), desc="Generating"):
            try:
                res = future.result()
                if res:
                    srcs, waves = res
                    # 合并结果
                    # 注意：字典 update 在单线程主进程中是安全的
                    all_sources.update(srcs)
                    all_waveforms.extend(waves)
            except Exception as e:
                print(f"Task Failed: {e}")

    # 3. 保存元数据
    print("Saving metadata...")
    
    # 保存 Grid Points
    grid_dict = {str(p["idx"]): p for p in grid_points}
    with open(os.path.join(GenConfig.OUTPUT_DIR, "..", "grid_points.json"), "w") as f:
        json.dump(grid_dict, f, indent=2)
        
    # 保存 Sources
    with open(os.path.join(GenConfig.OUTPUT_DIR, "..", "sources.json"), "w") as f:
        json.dump(all_sources, f, indent=2)
        
    # 保存 Waveforms
    with open(os.path.join(GenConfig.OUTPUT_DIR, "..", "waveforms.json"), "w") as f:
        json.dump(all_waveforms, f, indent=2)
        
    # 注意：receivers.json 不再需要，因为坐标在 waveforms.json 里
    
    print(f"All Done! Data saved to {GenConfig.OUTPUT_DIR}/..")

if __name__ == "__main__":
    # Windows/MacOS 下 multiprocessing 需要这个保护
    multiprocessing.freeze_support()
    main()


import obspy
from obspy.clients.fdsn import Client
import os
import numpy as np
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 1. 手动录入 GCMT 列表 (保持不变) =================
MANUAL_EVENTS = [
    {"id": "200506121541A", "time": "2005-06-12T15:41:50.7", "lat": 33.63, "lon": -116.59, "depth": 14.0, "mw": 5.2, "expo": 24, "tensor": [-0.025, -0.601, 0.626, 0.280, 0.410, -0.239]},
    {"id": "200506162053A", "time": "2005-06-16T20:53:29.6", "lat": 34.09, "lon": -116.99, "depth": 12.0, "mw": 4.9, "expo": 23, "tensor": [0.990, -2.380, 1.390, 1.420, 0.362, 0.791]},
    {"id": "200509020127A", "time": "2005-09-02T01:27:23.5", "lat": 33.23, "lon": -115.52, "depth": 12.0, "mw": 5.1, "expo": 23, "tensor": [-0.646, -4.730, 5.380, 1.980, -0.548, 3.690]},
    {"id": "200903241155A", "time": "2009-03-24T11:55:46.2", "lat": 33.31, "lon": -115.66, "depth": 12.0, "mw": 5.0, "expo": 23, "tensor": [-0.522, -2.120, 2.640, -0.204, -2.320, 1.460]},
    {"id": "201007072353A", "time": "2010-07-07T23:53:37.5", "lat": 33.41, "lon": -116.51, "depth": 24.1, "mw": 5.5, "expo": 24, "tensor": [-0.392, -2.330, 2.720, 0.217, 0.207, 0.304]},
    {"id": "201606100804B", "time": "2016-06-10T08:04:41.9", "lat": 33.51, "lon": -116.48, "depth": 24.7, "mw": 5.3, "expo": 24, "tensor": [-0.104, -0.775, 0.879, 0.371, 0.185, -0.436]},
    {"id": "202004040153B", "time": "2020-04-04T01:53:24.4", "lat": 33.57, "lon": -116.42, "depth": 24.8, "mw": 5.0, "expo": 23, "tensor": [-0.739, -2.440, 3.170, 2.280, 1.100, -2.020]},
    {"id": "202005102207A", "time": "2020-05-10T22:07:46.7", "lat": 33.39, "lon": -115.97, "depth": 18.8, "mw": 4.8, "expo": 23, "tensor": [0.298, -1.650, 1.360, -0.408, 0.594, -0.054]},
    {"id": "202010010031A", "time": "2020-10-01T00:31:29.5", "lat": 33.04, "lon": -115.60, "depth": 12.0, "mw": 5.0, "expo": 23, "tensor": [-0.570, -1.880, 2.450, -0.422, -0.909, 3.340]},
    {"id": "202106051755B", "time": "2021-06-05T17:56:01.2", "lat": 33.14, "lon": -115.60, "depth": 12.0, "mw": 5.3, "expo": 24, "tensor": [-0.080, -0.682, 0.763, -0.074, -0.218, 0.833]},
    {"id": "202504141708A", "time": "2025-04-14T17:08:33.0", "lat": 33.09, "lon": -116.57, "depth": 22.9, "mw": 5.3, "expo": 24, "tensor": [-0.130, -0.851, 0.980, -0.246, -0.043, 0.049]}
]

# ================= 2. 配置区域 =================
class Config:
    # 搜索配置
    NETWORK_CODE = "*"       # 搜索所有台网
    # NETWORK_CODE = "CI"        # 建议先只用 CI，如果不够再开 *，CI 数据最全
    SEARCH_RADIUS_DEG = 2.0    # 2.0度 (~220km)
    
    # 【核心配置】目标波形数量
    TARGET_STATIONS_PER_EVENT = 30  # 每个地震，下载到 30 个台站就收手
    
    # 通道配置 (宽频+短周期)
    CHANNELS = "HH*,BH*,EH*,SH*" 
    
    # 多线程
    MAX_WORKERS = 11  # 你要求的 12 线程并发
    
    # 波形处理 (Sim-to-Real 核心：0.25Hz)
    SAMPLE_RATE = 2.0
    POINTS = 512
    FREQ_MIN = 0.02
    FREQ_MAX = 0.25
    DURATION = POINTS / SAMPLE_RATE
    
    OUTPUT_DIR = "./data/output_real_exhaustive"
    WAVE_DIR = os.path.join(OUTPUT_DIR, "waveforms")

# ================= 3. 工具函数 =================
def convert_gcmt_to_ned(vals, expo):
    scale = (10 ** expo) * 1e-7
    mrr, mtt, mpp, mrt, mrp, mtp = [x * scale for x in vals]
    return [mtt, mpp, mrr, -mtp, mrt, -mrp]

# ================= 4. 单个地震处理任务 (贪婪下载) =================
def process_event_task(event, event_idx):
    """
    1. 搜索所有台站
    2. 随机打乱
    3. 挨个下载，直到凑齐 TARGET_STATIONS_PER_EVENT
    """
    # 独立客户端，防止冲突
    client_scedc = Client("SCEDC")
    client_iris = Client("IRIS")
    
    origin_time = obspy.UTCDateTime(event['time'])
    ev_id = event['id']
    
    local_waves = []
    local_rec = {}
    
    # 1. 准备元数据
    mt_ned = convert_gcmt_to_ned(event['tensor'], event['expo'])
    g_idx = 1000 + event_idx
    m_idx = 0
    
    src_meta = {
        "mechanism_idx": m_idx, "moment_tensor": mt_ned,
        "magnitude": event['mw'], "origin_time": str(origin_time)
    }
    grid_meta = {
        "idx": g_idx, "lat": event['lat'], "lon": event['lon'], "depth_km": event['depth']
    }
    
    # 2. 获取周边所有台站
    # 优先从 SCEDC 获取台站列表
    stations_info = []
    try:
        # 获取 Station Inventory (只含元数据，很快)
        inventory = client_scedc.get_stations(
            network=Config.NETWORK_CODE,
            latitude=event['lat'], longitude=event['lon'],
            maxradius=Config.SEARCH_RADIUS_DEG,
            starttime=origin_time, endtime=origin_time+600,
            level="station"
        )
        # 提取台站信息
        for net in inventory:
            for sta in net:
                stations_info.append((net.code, sta))
    except:
        # 如果 SCEDC 列表获取失败，尝试 IRIS
        try:
            inventory = client_iris.get_stations(
                network=Config.NETWORK_CODE,
                latitude=event['lat'], longitude=event['lon'],
                maxradius=Config.SEARCH_RADIUS_DEG,
                starttime=origin_time, endtime=origin_time+600,
                level="station"
            )
            for net in inventory:
                for sta in net:
                    stations_info.append((net.code, sta))
        except Exception as e:
            return False, ev_id, [], {}, {}, {}, f"Station Search Failed: {e}"

    if not stations_info:
        return False, ev_id, [], {}, {}, {}, "No stations found nearby"

    # 3. 【关键】随机打乱台站顺序，避免卡在某个坏掉的子台网
    random.shuffle(stations_info)
    
    # 4. 贪婪下载循环
    success_cnt = 0
    
    # 遍历所有找到的台站
    for net_code, sta in stations_info:
        # 如果已经凑够了，提前结束
        if success_cnt >= Config.TARGET_STATIONS_PER_EVENT:
            break
            
        sta_code = sta.code
        st = None
        used_client = None
        
        # 轮询客户端下载波形
        for cli in [client_scedc, client_iris]:
            try:
                st = cli.get_waveforms(
                    network=net_code, station=sta_code, location="*", channel=Config.CHANNELS,
                    starttime=origin_time, endtime=origin_time + Config.DURATION + 60,
                    attach_response=True
                )
                if len(st) >= 3: 
                    used_client = cli
                    break
            except:
                continue
        
        if st is None: continue # 下载失败，尝试下一个台站
        
        try:
            # --- 预处理流程 ---
            st.merge(fill_value=0)
            st.remove_response(output="DISP", pre_filt=(0.01, 0.015, 0.6, 0.8), water_level=60)
            
            inv_resp = used_client.get_stations(
                network=net_code, station=sta_code, starttime=origin_time, level="response"
            )
            st.rotate('->ZNE', inventory=inv_resp)
            
            st_zne = obspy.Stream()
            for cha in ["Z", "N", "E"]:
                tr = st.select(component=cha)
                if len(tr)>0: st_zne += tr[0]
            if len(st_zne) != 3: continue
            
            st_zne.filter('bandpass', freqmin=Config.FREQ_MIN, freqmax=Config.FREQ_MAX, corners=4, zerophase=True)
            st_zne.resample(Config.SAMPLE_RATE)
            st_zne.trim(starttime=origin_time, endtime=origin_time + Config.DURATION, pad=True, fill_value=0)
            
            # 对齐
            valid_trace = True
            for tr in st_zne:
                # 如果缺数据太多(全是0)，或者点数不对，就丢弃
                if np.all(tr.data == 0): 
                    valid_trace = False
                    break
                    
                if tr.stats.npts > Config.POINTS: tr.data = tr.data[:Config.POINTS]
                elif tr.stats.npts < Config.POINTS:
                    tr.data = np.pad(tr.data, (0, Config.POINTS - tr.stats.npts), 'constant')
            
            if not valid_trace: continue

            # 保存
            filename = f"{event['id']}_{net_code}.{sta_code}.mseed"
            st_zne.write(os.path.join(Config.WAVE_DIR, filename), format="MSEED")
            
            local_waves.append({
                "grid_point_idx": g_idx, "mechanism_idx": m_idx,
                "receiver_code": f"{net_code}.{sta_code}",
                "receiver_lat": sta.latitude, "receiver_lon": sta.longitude,
                "waveform_file": os.path.join("waveforms", filename),
                "split": "test"
            })
            
            # 记录台站位置
            local_rec_key = f"{net_code}.{sta_code}"
            if local_rec_key not in local_rec:
                local_rec[local_rec_key] = {
                    "network": net_code, "station": sta_code,
                    "lat": sta.latitude, "lon": sta.longitude, "elevation": sta.elevation
                }
            
            success_cnt += 1
            
        except Exception:
            pass
    
    # 只要下载到了 >= 4 个台站，就算成功
    if success_cnt >= 4:
        return True, event['id'], local_waves, src_meta, grid_meta, local_rec, f"Success: {success_cnt}/{len(stations_info)} stations"
    else:
        return False, event['id'], [], {}, {}, {}, f"Not enough data ({success_cnt}/{len(stations_info)})"

# ================= 5. 主程序 =================
def run():
    os.makedirs(Config.WAVE_DIR, exist_ok=True)
    
    print(f"🚀 Starting Exhaustive Download for {len(MANUAL_EVENTS)} events...")
    print(f"Workers: {Config.MAX_WORKERS} | Target Stations/Event: {Config.TARGET_STATIONS_PER_EVENT}")
    
    all_waves = []
    all_src = {}
    all_grd = {}
    all_rec = {}
    success_events = 0
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {executor.submit(process_event_task, ev, i): ev['id'] for i, ev in enumerate(MANUAL_EVENTS)}
        
        for future in tqdm(as_completed(futures), total=len(MANUAL_EVENTS), desc="Total Progress"):
            is_ok, eid, waves, src, grd, recs, msg = future.result()
            
            if is_ok:
                all_waves.extend(waves)
                g_id = str(grd['idx'])
                all_grd[g_id] = grd
                if g_id not in all_src: all_src[g_id] = {}
                all_src[g_id]["0"] = src
                all_rec.update(recs)
                success_events += 1
                tqdm.write(f"  [OK] {eid}: {msg}")
            else:
                tqdm.write(f"  [Fail] {eid}: {msg}")

    if success_events > 0:
        print(f"\nSuccessfully downloaded {success_events} events.")
        print("Saving metadata...")
        with open(os.path.join(Config.OUTPUT_DIR, "waveforms.json"), "w") as f:
            json.dump(all_waves, f, indent=2)
        with open(os.path.join(Config.OUTPUT_DIR, "sources.json"), "w") as f:
            json.dump(all_src, f, indent=2)
        with open(os.path.join(Config.OUTPUT_DIR, "grid_points.json"), "w") as f:
            json.dump(all_grd, f, indent=2)
        with open(os.path.join(Config.OUTPUT_DIR, "receivers.json"), "w") as f:
            json.dump(all_rec, f, indent=2)
        print(f"Done! Data in {Config.OUTPUT_DIR}")
    else:
        print("No valid events found.")

if __name__ == "__main__":
    # 忽略烦人的警告
    import warnings
    warnings.filterwarnings("ignore")
    run()
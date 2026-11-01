import json
import torch
import numpy as np
import os
import random
import obspy
from obspy.geodetics import gps2dist_azimuth
from torch.utils.data import Dataset
from .config import Config

class DynamicSeismicDataset(Dataset):
    def __init__(self, split='train', augment=False):
        self.cfg = Config
        self.split = split
        self.augment = augment
        
        print(f"[{split}] Loading metadata from {self.cfg.JSON_PATHS['waveforms']} ...")
        
        # 加载元数据
        with open(self.cfg.JSON_PATHS['grid'], 'r') as f:
            self.grids = json.load(f)
        with open(self.cfg.JSON_PATHS['source'], 'r') as f:
            self.sources = json.load(f)
        with open(self.cfg.JSON_PATHS['waveforms'], 'r') as f:
            all_waves = json.load(f)
            
        # 整理 Events
        self.events = {}
        for w in all_waves:
            if w.get('split') != split: continue
            
            # 唯一 Event ID
            # 注意：这里 waveform json 里的 key 是 'grid_point_idx'
            key = f"{w['grid_point_idx']}_{w['mechanism_idx']}"
            
            if key not in self.events:
                self.events[key] = {
                    # 【关键】统一内部使用的 Key 为 grid_idx 和 mech_idx
                    'grid_idx': str(w['grid_point_idx']),
                    'mech_idx': str(w['mechanism_idx']),
                    'waveforms': []
                }
            self.events[key]['waveforms'].append(w)
            
        self.event_keys = list(self.events.keys())
        print(f"[{split}] Loaded {len(self.event_keys)} unique events.")

    def __len__(self):
        return len(self.event_keys)

    def __getitem__(self, idx):
        # 1. 获取 Event ID
        key = self.event_keys[idx]
        evt_data = self.events[key]
        
        # 【修复】使用正确的 Key 读取
        g_idx = evt_data['grid_idx']
        m_idx = evt_data['mech_idx']
        
        # 2. 获取 Label (MT, Mw)
        src = self.sources[g_idx][m_idx]
        mt_raw = np.array(src['moment_tensor'], dtype=np.float32)
        
        # 计算震级
        m0 = np.sqrt(np.sum(mt_raw**2) / 2.0)
        mw = (2.0/3.0) * np.log10(m0 + 1e-10) - 6.07
        
        # 归一化 MT (Shape Label)
        mt_norm = mt_raw / (np.linalg.norm(mt_raw) + 1e-10)
        
        # 获取震源坐标 (用于计算几何)
        grid = self.grids[g_idx]
        src_lat = grid['lat']
        src_lon = grid['lon']
        
        # 兼容 depth 单位 (km vs m)
        if 'depth_km' in grid:
            src_dep_km = grid['depth_km']
        else:
            src_dep_km = grid.get('depth_m', 10000) / 1000.0

        # 3. 智能读取与选台策略 (Smart Station Selection)
        all_meta_list = evt_data['waveforms']
        valid_candidates = []
        
        for w_meta in all_meta_list:
            # 路径处理
            rel_path = w_meta['waveform_file']
            if rel_path.startswith("train/") and self.cfg.DATA_ROOT.endswith("train"):
                rel_path = rel_path.replace("train/", "", 1)
            full_path = os.path.join(self.cfg.DATA_ROOT, rel_path)
            
            if not os.path.exists(full_path): continue

            try:
                st = obspy.read(full_path)
                data = np.array([tr.data for tr in st])[:3, :self.cfg.SEQ_LEN]
                
                if data.shape != (3, self.cfg.SEQ_LEN): continue
                
                peak_amp = np.max(np.abs(data))
                
                if peak_amp > 1e-15: 
                    valid_candidates.append({
                        'data': data,
                        'meta': w_meta,
                        'score': peak_amp
                    })
            except:
                continue

        # 如果没有有效波形，返回空
        if not valid_candidates:
            return self._empty_sample()

        # 选台逻辑
        if self.augment:
            # 训练：随机选
            num_pick = random.randint(4, min(len(valid_candidates), self.cfg.MAX_STATIONS))
            selected_items = random.sample(valid_candidates, num_pick)
        else:
            # 测试：选最好的 (振幅最大)
            valid_candidates.sort(key=lambda x: x['score'], reverse=True)
            selected_items = valid_candidates[:self.cfg.MAX_STATIONS]

        # 4. 构建 Tensor
        wav_tensor = torch.zeros(self.cfg.MAX_STATIONS, 3, self.cfg.SEQ_LEN, dtype=torch.float32)
        geo_tensor = torch.zeros(self.cfg.MAX_STATIONS, self.cfg.GEO_DIM, dtype=torch.float32)
        mask = torch.ones(self.cfg.MAX_STATIONS, dtype=torch.bool) 
        
        event_max = 0.0
        
        for i, item in enumerate(selected_items):
            data = item['data']
            w_meta = item['meta']
            
            curr_max = item['score']
            if curr_max > event_max: event_max = curr_max
            
            # Wave
            wav_tensor[i] = torch.from_numpy(data)
            
            # Geometry
            rec_lat = w_meta.get('receiver_lat', 0.0)
            rec_lon = w_meta.get('receiver_lon', 0.0)
            dist, az, _ = gps2dist_azimuth(src_lat, src_lon, rec_lat, rec_lon)
            
            geo_tensor[i, 0] = np.sin(np.deg2rad(az))
            geo_tensor[i, 1] = np.cos(np.deg2rad(az))
            geo_tensor[i, 2] = (dist/1000.0) / self.cfg.MAX_DIST_KM
            geo_tensor[i, 3] = src_dep_km / 30.0
            
            mask[i] = False

        # 5. 归一化 & 增强
        if event_max > 1e-15:
            wav_tensor /= event_max
            log_amp = np.log10(event_max + 1e-10)
            
            if self.augment:
                noise = torch.randn_like(wav_tensor) * self.cfg.NOISE_LEVEL
                valid_indices = ~mask
                wav_tensor[valid_indices] += noise[valid_indices]
                
                if random.random() < self.cfg.AUGMENT_PROB:
                    wav_tensor = -wav_tensor
                    mt_norm = -mt_norm
        else:
            log_amp = -9.0
            
        if torch.isnan(wav_tensor).any():
            wav_tensor = torch.zeros_like(wav_tensor)
            log_amp = -9.0

        return (wav_tensor, geo_tensor, 
                torch.tensor(mt_norm, dtype=torch.float32), 
                torch.tensor([mw], dtype=torch.float32), 
                torch.tensor([log_amp], dtype=torch.float32), 
                mask)

    def _empty_sample(self):
        w = torch.zeros(self.cfg.MAX_STATIONS, 3, self.cfg.SEQ_LEN)
        g = torch.zeros(self.cfg.MAX_STATIONS, self.cfg.GEO_DIM)
        m_mt = torch.zeros(6)
        m_mw = torch.zeros(1)
        m_amp = torch.tensor([-9.0])
        mask = torch.ones(self.cfg.MAX_STATIONS, dtype=torch.bool)
        return w, g, m_mt, m_mw, m_amp, mask
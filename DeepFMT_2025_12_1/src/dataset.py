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
        
        print(f"[{split}] Loading metadata...")
        with open(self.cfg.JSON_PATHS['grid'], 'r') as f: self.grids = json.load(f)
        with open(self.cfg.JSON_PATHS['source'], 'r') as f: self.sources = json.load(f)
        with open(self.cfg.JSON_PATHS['waveforms'], 'r') as f: all_waves = json.load(f)
            
        # Group by Event
        self.events = {}
        for w in all_waves:
            if w.get('split') != split: continue
            key = f"{w['grid_point_idx']}_{w['mechanism_idx']}"
            if key not in self.events: self.events[key] = {'waveforms': []}
            self.events[key]['waveforms'].append(w)
            
        self.event_keys = list(self.events.keys())
        print(f"[{split}] Loaded {len(self.event_keys)} unique events.")

    def __len__(self):
        return len(self.event_keys)

    def __getitem__(self, idx):
        key = self.event_keys[idx]
        evt_data = self.events[key]
        
        # 获取 Label
        first = evt_data['waveforms'][0]
        g_idx, m_idx = str(first['grid_point_idx']), str(first['mechanism_idx'])
        
        src = self.sources[g_idx][m_idx]
        mt_raw = np.array(src['moment_tensor'], dtype=np.float32)




        m0 = np.sqrt(np.sum(mt_raw**2)/2.0)
        mw = (2/3)*np.log10(m0 + 1e-10) - 6.07


        
        # # 直接拿 json 里面的真实震级，不自己算：
        # mw = src.get('magnitude')
        # # 如果非要自己算，使用正确的数学公式：
        # if mw is None:
        #      m0 = np.sqrt((np.sum(mt_raw[:3]**2) + 2 * np.sum(mt_raw[3:]**2)) / 2.0)
        #      mw = (2.0/3.0) * np.log10(m0 + 1e-10) - 6.07


        mt_norm = mt_raw / (np.linalg.norm(mt_raw) + 1e-10)
        
        # 动态选台 (Station Dropout)
        waves = evt_data['waveforms']
        if self.augment and random.random() < self.cfg.STATION_DROPOUT_PROB:
            k = random.randint(4, min(len(waves), self.cfg.MAX_STATIONS))
            selected = random.sample(waves, k)
        else:
            limit = min(len(waves), self.cfg.MAX_STATIONS)
            selected = waves[:limit]
            
        # 构建 Tensor (Padding)
        wav_tensor = torch.zeros(self.cfg.MAX_STATIONS, 3, self.cfg.SEQ_LEN, dtype=torch.float32)
        geo_tensor = torch.zeros(self.cfg.MAX_STATIONS, self.cfg.GEO_DIM, dtype=torch.float32)
        mask = torch.ones(self.cfg.MAX_STATIONS, dtype=torch.bool) # True = Padding
        
        # 震源位置
        grid = self.grids[g_idx]
        event_max = 0.0
        
        for i, w_meta in enumerate(selected):
            try:
                # 读取波形
                fp = os.path.join(self.cfg.DATA_ROOT, w_meta['waveform_file'])
                st = obspy.read(fp)
                data = np.array([tr.data for tr in st])[:3, :self.cfg.SEQ_LEN]
                
                curr_max = np.max(np.abs(data))
                if curr_max > event_max: event_max = curr_max
                
                wav_tensor[i, :, :data.shape[1]] = torch.from_numpy(data)
                
                # 计算几何 (直接从 meta 读坐标)
                rec_lat, rec_lon = w_meta['receiver_lat'], w_meta['receiver_lon']
                dist, az, _ = gps2dist_azimuth(grid['lat'], grid['lon'], rec_lat, rec_lon)
                dep = grid.get('depth_km', grid.get('depth_m', 10000)/1000.0)
                
                geo_tensor[i, 0] = np.sin(np.deg2rad(az))
                geo_tensor[i, 1] = np.cos(np.deg2rad(az))
                geo_tensor[i, 2] = (dist/1000.0) / self.cfg.MAX_DIST_KM
                geo_tensor[i, 3] = dep / 30.0
                
                mask[i] = False # 标记有效
            except: pass
            
        # 归一化 & 增强
        if event_max > 0:
            wav_tensor /= event_max
            log_amp = np.log10(event_max + 1e-10)
            
            if self.augment:
                # 随机生成轻微的噪声 (在 0 到设定的最大噪声水平之间)
                actual_noise_level = random.uniform(0.0, self.cfg.NOISE_LEVEL)
                noise = torch.randn_like(wav_tensor) * actual_noise_level
                wav_tensor.masked_fill_(mask.unsqueeze(1).unsqueeze(2), 0.0)
                wav_tensor += noise * (~mask.unsqueeze(1).unsqueeze(2))
                
                # 极性翻转
                if random.random() < self.cfg.AUGMENT_PROB:
                    wav_tensor = -wav_tensor
                    mt_norm = -mt_norm
        else:
            log_amp = -9.0
        
        if torch.isnan(wav_tensor).any() or torch.isinf(wav_tensor).any():
            # 如果发现坏数据，直接返回全 0
            wav_tensor = torch.zeros_like(wav_tensor)
            log_amp = -9.0
            # 也可以在这里 print 报警
        return wav_tensor, geo_tensor, torch.tensor(mt_norm).float(), torch.tensor([mw]).float(), torch.tensor([log_amp]).float(), mask
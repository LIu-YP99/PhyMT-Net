import json
import random
import os
import sys
from collections import defaultdict

# 引入路径以便读取 Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def split_dataset():
    path = Config.JSON_PATHS['waveforms']
    print(f"Reading {path}...")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"Total waveform traces: {len(data)}")
    
    # 1. 按事件分组 (Group by Unique Event ID)
    # 必须确保同一个地震的所有台站波形都在同一个集里，严禁跨集！
    events_map = defaultdict(list)
    for i, w in enumerate(data):
        # 唯一标识符：网格点ID + 机制ID
        event_id = f"{w['grid_point_idx']}_{w['mechanism_idx']}"
        events_map[event_id].append(i) # 记录索引
        
    unique_event_ids = list(events_map.keys())
    total_events = len(unique_event_ids)
    print(f"Total unique events: {total_events}")
    
    # 2. 随机打乱事件 ID
    random.seed(42)
    random.shuffle(unique_event_ids)
    
    # 3. 计算划分点 (8:1:1)
    n_train = int(total_events * 0.8)
    n_val = int(total_events * 0.1)
    # n_test = 剩余
    
    train_ids = set(unique_event_ids[:n_train])
    val_ids = set(unique_event_ids[n_train:n_train+n_val])
    test_ids = set(unique_event_ids[n_train+n_val:])
    
    # 4. 回写标签
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    for eid in unique_event_ids:
        # 确定该事件属于哪个集
        if eid in train_ids:
            split_label = 'train'
        elif eid in val_ids:
            split_label = 'val'
        else:
            split_label = 'test'
            
        # 将该事件下的所有波形都标记为这个 label
        for idx in events_map[eid]:
            data[idx]['split'] = split_label
            counts[split_label] += 1
            
    # 5. 保存
    print("Saving updated waveforms.json...")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Split Summary (Waveforms):")
    print(f"  Train: {counts['train']}")
    print(f"  Val  : {counts['val']}")
    print(f"  Test : {counts['test']}")
    print(f"  Total: {sum(counts.values())}")

if __name__ == "__main__":
    split_dataset()
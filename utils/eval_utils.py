import os
import numpy as np
import cv2
import trimesh
import re
from glob import glob
from sklearn.neighbors import NearestNeighbors

# ==============================================================================
# 1. MAE (法线) 计算
# ==============================================================================

def load_normal_from_png(path):
    """
    加载法线图，支持 8-bit 和 16-bit，归一化到 [-1, 1]
    """
    # 读取图像
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    
    # 转换为 float 并归一化到 [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    
    # 处理通道 (BGR -> RGB)
    if len(img.shape) == 3:
        if img.shape[2] >= 3:
            img = img[:, :, :3] # 丢弃 Alpha
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # [0, 1] -> [-1, 1]
    normal = img * 2.0 - 1.0
    
    # 计算模长并生成 Mask
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    
    # 有效性判断：模长接近 1 的才是有效法线 (过滤背景)
    valid_mask = (norm.squeeze() > 0.1) & (norm.squeeze() < 10.0)
    
    # 归一化 (防止除零)
    normal = np.where(norm > 1e-6, normal / norm, 0)
    
    return normal, valid_mask

def extract_number(filename):
    """
    使用正则表达式从文件名中提取第一个连续数字序列。
    例如: 
    '00001.png' -> 1
    '0001.png'  -> 1
    'frame_01.jpg' -> 1
    """
    basename = os.path.basename(filename)
    # 寻找数字
    match = re.search(r'\d+', basename)
    if match:
        return int(match.group())
    return None

def compute_mae_from_files(pred_dir, gt_dir):
    """
    遍历文件夹计算 MAE (基于提取的数字ID进行匹配)
    """
    # 搜索文件
    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")) + glob(os.path.join(gt_dir, "*.jpg")))
    
    if not pred_files:
        print(f"[MAE] 错误: 预测目录为空: {pred_dir}")
        return None
    if not gt_files:
        print(f"[MAE] 错误: GT 目录为空: {gt_dir}")
        return None

    # --- 构建 GT 索引映射 ---
    # 格式: { 1: '/path/to/0001.png', 2: '/path/to/0002.png' }
    gt_map = {}
    for f in gt_files:
        fid = extract_number(f)
        if fid is not None:
            gt_map[fid] = f
    
    print(f"[MAE] 预测文件: {len(pred_files)} 张")
    print(f"[MAE] GT文件:   {len(gt_files)} 张 (ID范围: {min(gt_map.keys()) if gt_map else 'N/A'} - {max(gt_map.keys()) if gt_map else 'N/A'})")

    all_angles = []
    missing_gt_count = 0
    empty_mask_count = 0
    matched_count = 0

    for pred_path in pred_files:
        # 1. 提取预测文件的数字 ID (00001 -> 1)
        pred_id = extract_number(pred_path)
        
        if pred_id is None:
            print(f"  [警告] 无法从文件名提取数字: {os.path.basename(pred_path)}")
            continue

        # 2. 从 GT Map 中查找对应的 ID (0001 -> 1)
        gt_path = gt_map.get(pred_id)

        if gt_path is None:
            # 仅打印前几个缺失的警告
            if missing_gt_count < 3:
                print(f"  [匹配失败] Pred: {os.path.basename(pred_path)} (ID={pred_id}) -> 在 GT 中找不到 ID={pred_id}")
            missing_gt_count += 1
            continue

        matched_count += 1
        
        try:
            pred_normal, pred_mask = load_normal_from_png(pred_path)
            gt_normal, gt_mask = load_normal_from_png(gt_path)
            
            # 尺寸对齐
            if pred_normal.shape[:2] != gt_normal.shape[:2]:
                h, w = pred_normal.shape[:2]
                gt_normal = cv2.resize(gt_normal, (w, h), interpolation=cv2.INTER_LINEAR)
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            # 联合 Mask
            valid_mask = pred_mask & gt_mask
            
            if valid_mask.sum() == 0:
                if empty_mask_count < 3:
                    print(f"  [无效区域] {os.path.basename(pred_path)} 与 GT 无重叠有效区域")
                empty_mask_count += 1
                continue
            
            # 计算误差
            pred_n = pred_normal[valid_mask]
            gt_n = gt_normal[valid_mask]
            
            dot = np.sum(pred_n * gt_n, axis=-1)
            dot = np.clip(dot, -1.0, 1.0)
            
            angles = np.arccos(np.abs(dot)) 
            angles_deg = np.degrees(angles)
            all_angles.extend(angles_deg)
            
        except Exception as e:
            print(f"  [Error] 处理 {os.path.basename(pred_path)} 失败: {e}")

    # 打印统计信息
    if matched_count == 0:
        print("\n[致命错误] 未成功匹配任何图片！请检查文件名中的数字是否对应。")
        print(f"  Pred 示例: {os.path.basename(pred_files[0])} -> ID={extract_number(pred_files[0])}")
        print(f"  GT   示例: {os.path.basename(gt_files[0])} -> ID={extract_number(gt_files[0])}")
        return None

    if missing_gt_count > 0:
        print(f"  [统计] {missing_gt_count} 张图片未找到对应 GT (已跳过)。")
    
    if not all_angles:
        print("  [统计] 所有匹配图片的有效区域均为空。")
        return None
        
    return np.mean(all_angles)

# ==============================================================================
# 2. CD (Chamfer Distance) 计算
# ==============================================================================

def load_mesh_vertices(path):
    """使用 trimesh 加载顶点"""
    try:
        mesh = trimesh.load(path, force='mesh', process=False)
    except:
        try:
            mesh = trimesh.load(path, process=False)
        except Exception as e:
            raise ValueError(f"Trimesh 加载失败: {e}")
    
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0: return np.array([])
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        
    if hasattr(mesh, 'vertices'):
        return np.array(mesh.vertices)
    return np.array([])

def compute_cd_from_files(pred_path, gt_path):
    """CD 计算"""
    # 1. 加载
    try:
        pred_pts = load_mesh_vertices(pred_path)
        gt_pts = load_mesh_vertices(gt_path)
    except Exception as e:
        print(f"  [CD Error] 加载模型失败: {e}")
        return None
    
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        print(f"  [CD Error] 点云顶点数为 0")
        return None
        
    print(f"  Pred 顶点数: {len(pred_pts)}")
    print(f"  GT   顶点数: {len(gt_pts)}")

    # 2. 构建 KD-Tree
    nn_gt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1).fit(gt_pts)
    nn_pred = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1).fit(pred_pts)
    
    # 3. 查询
    dists_p2g, _ = nn_gt.kneighbors(pred_pts)
    dists_g2p, _ = nn_pred.kneighbors(gt_pts)
    
    # 4. 过滤 (max_dist=10)
    max_dist = 10.0
    valid_p2g = dists_p2g[dists_p2g < max_dist]
    valid_g2p = dists_g2p[dists_g2p < max_dist]
    
    mean_p2g = np.mean(valid_p2g) if len(valid_p2g) > 0 else 0
    mean_g2p = np.mean(valid_g2p) if len(valid_g2p) > 0 else 0
    
    # 5. 计算 (x100)
    cd = (mean_p2g + mean_g2p) / 2.0
    cd *= 100.0
    
    return cd

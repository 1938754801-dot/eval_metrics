"""
python eval_metrics.py \
    -s data/SMVP3D/david \
    -m exp/david_20260129124848 \
    --iteration 15000 \
    --mesh_name poisson_mesh_8_pruned.ply
"""
import argparse
import os
import sys
# 引入刚才写的纯粹工具函数
from utils.eval_utils import compute_mae_from_files, compute_cd_from_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolGS Evaluation (No Open3D, No Gaussian Loading)")
    
    # 必须的参数
    parser.add_argument("--pred_dir", "-m", type=str, required=True, help="实验输出目录 (例如 exp/snail_xxx)")
    parser.add_argument("--data_dir", "-s", type=str, required=True, help="数据集目录 (例如 data/SMVP3D/snail)")
    
    # 可选参数
    parser.add_argument("--iteration", type=int, default=15000, help="迭代次数 (用于寻找 normal 文件夹)")
    parser.add_argument("--mesh_name", type=str, default="poisson_mesh_8_pruned.ply", help="Mesh 文件名")
    parser.add_argument("--skip_mae", action="store_true")
    parser.add_argument("--skip_cd", action="store_true")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"评估")
    print(f"Pred Dir: {args.pred_dir}")
    print(f"Data Dir: {args.data_dir}")
    print("="*60)
    
    # --------------------------------------------------------------------------
    # 1. 计算 MAE
    # --------------------------------------------------------------------------
    if not args.skip_mae:
        pred_normal_path = os.path.join(args.pred_dir, "train", str(args.iteration), "normal")
        gt_normal_path = os.path.join(args.data_dir, "normal")
        
        print("\n[1/2] Calculating MAE...")
        if os.path.exists(pred_normal_path) and os.path.exists(gt_normal_path):
            mae = compute_mae_from_files(pred_normal_path, gt_normal_path)
            if mae is not None:
                print(f"  ✓ MAE: {mae:.4f}°")
                # 保存结果
                with open(os.path.join(args.pred_dir, "metric_mae.txt"), "w") as f:
                    f.write(f"MAE: {mae}\n")
            else:
                print("  MAE 计算失败 (无有效数据)")
        else:
            print("  [跳过] 找不到法线文件夹:")
            if not os.path.exists(pred_normal_path): print(f"    Missing: {pred_normal_path}")
            if not os.path.exists(gt_normal_path):   print(f"    Missing: {gt_normal_path}")

    # --------------------------------------------------------------------------
    # 2. 计算 CD
    # --------------------------------------------------------------------------
    if not args.skip_cd:
        # 自动拼接路径
        pred_mesh_path = os.path.join(args.pred_dir, args.mesh_name)
        
        # 尝试寻找 GT Mesh
        scene_name = os.path.basename(os.path.normpath(args.data_dir))
        gt_mesh_path = os.path.join(args.data_dir, f"{scene_name}.obj")
        if not os.path.exists(gt_mesh_path):
             gt_mesh_path = os.path.join(args.data_dir, f"{scene_name}.ply")
        
        print("\n[2/2] Calculating Chamfer Distance...")
        if os.path.exists(pred_mesh_path) and os.path.exists(gt_mesh_path):
            print(f"  Pred: {os.path.basename(pred_mesh_path)}")
            print(f"  GT:   {os.path.basename(gt_mesh_path)}")
            
            try:
                cd = compute_cd_from_files(pred_mesh_path, gt_mesh_path)
                print(f"  ✓ CD:  {cd:.4f} (scale x100)")
                
                # 保存结果
                with open(os.path.join(args.pred_dir, "metric_cd.txt"), "w") as f:
                    f.write(f"CD: {cd}\n")
            except Exception as e:
                print(f"  CD 计算出错: {e}")
        else:
            print("  [跳过] 找不到 Mesh 文件:")
            if not os.path.exists(pred_mesh_path): print(f"    Missing: {pred_mesh_path}")
            if not os.path.exists(gt_mesh_path):   print(f"    Missing: {gt_mesh_path}")

    print("\nDone.")
使用方式：
1、将 eval_utils.py 存放在 utils 文件夹下
2、将 eval_metrics.py 存放在PolGS的根目录中，通过以下指令进行调用

python eval_metrics.py \
    -s data/SMVP3D/david \
    -m exp/david_20260129124848 \
    --iteration 15000 \
    --mesh_name poisson_mesh_8_pruned.ply

需注意调用时针对不同数据集、训练iter，以及存放在训练结果中point_cloud的mesh文件进行修改

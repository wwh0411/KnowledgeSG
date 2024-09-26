import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
path = '/mnt/workspace/wenhaowang/EasyFedLLM/output/fl-origin500_30000_fedavg_c30s5_i100_b1a1_l512_r16a32_20240321131455/training_loss.npy'
path = '/mnt/workspace/wenhaowang/EasyFedLLM/output/alpaca-gpt4_10000_fedavg_c50s5_i10_b12a1_l512_r16a32_20240225175129/training_loss.npy'
path = '/mnt/workspace/wenhaowang/EasyFedLLM/output/fl+dp-origin500_30000_dp_c30s5_i100_b1a1_l512_r16a32_20240321131351/training_loss.npy'
path = '/mnt/workspace/wenhaowang/EasyFedLLM/output/fl+dp-origin500change-collator-newarg_30000_dp_c30s3_i10_b1a1_l512_r16a32_20240407221121/training_loss.npy'
path = '/mnt/workspace/wenhaowang/EasyFedLLM/output/fl+dp-origin500changetrainer_30000_dp_c30s5_i100_b1a1_l512_r16a32_20240407175503/training_loss.npy'
path = '/mnt/workspace/wenhaowang/EasyFedLLM/output/fl+dp-origin500change-collator_30000_dp_c30s5_i100_b1a1_l512_r16a32_20240407205204/training_loss.npy'
data = np.load(path)
print(data.shape)
print(data[0])
# 绘制数据
# plt.figure(figsize=(8, 6))
# plt.plot(data)
# plt.xlabel('X轴标签')
# plt.ylabel('Y轴标签')
# plt.title('数据可视化')
# plt.grid(True)

# # 保存图形到文件
# plt.savefig('plot.png')

# # 显示图形（可选）
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 任务名称
tasks = ['Rating Count', 'Top5 Movie', 'Additional Analysis', 'ALS', 'Other Model']

# 机型
models = ['Cloud Cluster', 'Cloud Local', 'i5-13600k + 32g', 'M1 pro + 16g']

# 每个机型运行每个任务的时间
times = np.array([[53, 56, 48, 18],
                  [78, 102, 100, 34],
                  [652, 1579, 283, 148],
                  [270, 408, 117, 77],
                  [1042, 1666, 202, 0]])

times= times.T
# 创建一个图形窗口
fig, ax = plt.subplots(figsize=(10, 6))

# 设置每个机型的宽度
width = 0.2

# 绘制四种机型的四个任务的时间
for i in range(len(models)):
    ax.bar(np.arange(len(tasks)) + i*width, times[i], width, label=models[i])

# 设置图例
ax.legend()

# 设置x轴和y轴的标签

ax.set_ylabel('second')

# 设置x轴刻度
ax.set_xticks(np.arange(len(tasks))+0.3)
ax.set_xticklabels(tasks)

# 显示图形
plt.show()
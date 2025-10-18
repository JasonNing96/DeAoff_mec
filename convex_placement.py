# 修复后的优化模型
import cvxpy as cp
import numpy as np
import random
import matplotlib.pyplot as plt 
# 固定随机种子，以保证结果的可重复性
random.seed(42)
np.random.seed(42)

# 函数补充
# 绘制任务调度甘特图
def plot_gantt_chart(t_start, t_vm, z, num_nodes):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_nodes))
    for v in range(len(t_start)):
        for m in range(num_nodes):
            if z[v, m] > 0.5:  # 任务分配到节点 m
                plt.barh(m, t_vm[v, m], left=t_start[v], color=colors[m], edgecolor="black", label=f"Task {v}" if m == 0 else None)
    
    plt.yticks(range(num_nodes), [f"Node {m}" for m in range(num_nodes)])
    plt.xlabel("Time")
    plt.ylabel("Nodes")
    plt.title("Task Scheduling Gantt Chart")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# 贪婪算法作为基线对比
def greedy_scheduling(t_vm, c_mm, r_vm, C_m, E):
    num_tasks, num_nodes = t_vm.shape
    t_start = np.zeros(num_tasks)  # 每个任务的开始时间
    z = np.zeros((num_tasks, num_nodes))  # 任务分配

    node_resources = C_m.copy()  # 每个节点的剩余资源
    task_done = [False] * num_tasks  # 任务完成标记

    # 在 greedy_scheduling 函数中
    for v in range(num_tasks):
        # 查找可以执行任务的节点
        for m in range(num_nodes):
            if np.all(r_vm[v, m] <= node_resources[m]):
                z[v, m] = 1
                node_resources[m] -= r_vm[v, m]

                # 确定任务开始时间，考虑依赖关系
                predecessors = [t_start[u] + t_vm[u, m] for u, v_prime in E if v_prime == v]
                t_start[v] = max(predecessors) if predecessors else 0  # 处理空序列
                break

    # 计算总工期
    T_greedy = max(t_start[v] + t_vm[v, m] for v in range(num_tasks) for m in range(num_nodes) if z[v, m] > 0.5)
    return T_greedy, t_start, z

# 第一部分：定义任务依赖图生成函数
# 生成 GE 结构（高斯消元结构）
def generate_ge_dag(num_tasks):
    edges = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if random.random() < 0.5:  # 随机决定是否添加依赖关系
                edges.append((i, j))
    return edges

# 生成 FFT 结构（快速傅里叶变换结构）
def generate_fft_dag(num_tasks):
    edges = []
    levels = int(np.log2(num_tasks))
    for l in range(levels):
        step = 2**(l + 1)
        for i in range(0, num_tasks, step):
            for j in range(i, i + step // 2):
                if j + step // 2 < num_tasks:
                    edges.append((j, j + step // 2))
    return edges

# 选择 DAG 结构
# 选择 DAG 结构的函数
def generate_dag(num_tasks, structure):
    if structure == "GE":
        return generate_ge_dag(num_tasks)
    elif structure == "FFT":
        return generate_fft_dag(num_tasks)
    else:
        raise ValueError("Unsupported DAG structure")

structure = "FFT"  # 可选 "GE" 或 "FFT"
num_tasks = 50  # 任务数量
num_nodes = 10
E = generate_dag(num_tasks, structure)

# 打印生成的 DAG
print("生成的任务依赖关系:", E)


# 每个任务在每个节点上的执行时间 (num_tasks x num_nodes)
t_vm = np.random.randint(2, 10, size=(num_tasks, num_nodes))  # 随机生成 2 到 10 的执行时间

# 节点之间的通信延迟 (num_nodes x num_nodes)
c_mm = np.random.randint(1, 5, size=(num_nodes, num_nodes))  # 随机生成 1 到 5 的通信延迟
np.fill_diagonal(c_mm, 0)  # 自身通信延迟为 0

# 节点资源容量
C_m = np.random.randint(10, 20, size=num_nodes)  # 随机生成每个节点的资源容量

# 每个任务在每个节点上的资源需求 (num_tasks x num_nodes)
r_vm = np.random.randint(1, 5, size=(num_tasks, num_nodes))  # 随机生成 1 到 5 的资源需求
# t_vm = np.array([[3, 5, 8], [2, 6, 7], [4, 3, 6], [5, 4, 9], [7, 2, 4]])
# c_mm = np.array([[0, 2, 4], [2, 0, 3], [4, 3, 0]])
# a_vv = np.array([[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 3], [0, 0, 0, 0, 0]])
# C_m = np.array([10, 15, 12])
# r_vm = np.array([[2, 3, 5], [1, 4, 3], [3, 2, 4], [4, 5, 6], [2, 3, 2]])
# E = [(0, 1), (1, 2), (2, 3), (3, 4)]

# 决策变量定义
z = cp.Variable((num_tasks, num_nodes), boolean=True)
t = cp.Variable(num_tasks)
T = cp.Variable()
x = {
    (v, v_prime, m, m_prime): cp.Variable(boolean=True)
    for v, v_prime in E
    for m in range(num_nodes)
    for m_prime in range(num_nodes)
}

# 目标函数
objective = cp.Minimize(T)

# 约束条件
constraints = []

# 任务分配和资源约束
for v in range(num_tasks):
    constraints.append(cp.sum(z[v, :]) == 1)
for m in range(num_nodes):
    constraints.append(cp.sum(cp.multiply(z[:, m], r_vm[:, m])) <= C_m[m])

# 时间依赖约束
for (v, v_prime) in E:
    communication_delay = cp.sum([
        x[(v, v_prime, m, m_prime)] * c_mm[m, m_prime]
        for m in range(num_nodes) for m_prime in range(num_nodes)
    ])
    constraints.append(t[v_prime] >= t[v] + cp.sum(cp.multiply(z[v, :], t_vm[v, :])) + communication_delay)
    for m in range(num_nodes):
        for m_prime in range(num_nodes):
            constraints.append(x[(v, v_prime, m, m_prime)] <= z[v, m])
            constraints.append(x[(v, v_prime, m, m_prime)] <= z[v_prime, m_prime])
            constraints.append(x[(v, v_prime, m, m_prime)] >= z[v, m] + z[v_prime, m_prime] - 1)

# 工期约束
for v in range(num_tasks):
    constraints.append(T >= t[v] + cp.sum(cp.multiply(z[v, :], t_vm[v, :])))
    constraints.append(t[v] >= 0)

constraints.append(T >= 0)

# 求解模型
problem = cp.Problem(objective, constraints)
solution = problem.solve(solver=cp.GUROBI)
z_rounded = np.round(z.value, decimals=2)
# 输出结果
print("总工期 T:", T.value,z_rounded)
# print("任务开始时间 t:", t.value)
# print("任务分配 z:", z_rounded)

# 调用绘图函数
# plot_gantt_chart(t.value, t_vm, np.round(z.value, 2), num_nodes)

# greedy_T, greedy_t_start, greedy_z = greedy_scheduling(t_vm, c_mm, r_vm, C_m, E)
# print("贪婪算法总工期:", greedy_T,greedy_z)
# print("贪婪算法任务开始时间:", greedy_t_start)
# print("贪婪算法任务分配:", greedy_z)

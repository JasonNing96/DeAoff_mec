import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import matplotlib.pyplot as plt

# 固定随机种子
random.seed(42)
np.random.seed(42)

# 绘制任务调度甘特图
def plot_gantt_chart(t_finish, t_vm, z, num_nodes):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_nodes))
    for v in range(len(t_finish)):
        for m in range(num_nodes):
            if z[v, m] > 0.5:
                start_time = t_finish[v] - t_vm[v, m]
                plt.barh(m, t_vm[v, m], left=start_time, color=colors[m], edgecolor="black")
    
    plt.yticks(range(num_nodes), [f"Node {m}" for m in range(num_nodes)])
    plt.xlabel("Time")
    plt.ylabel("Nodes")
    plt.title("Task Scheduling Gantt Chart")
    plt.grid(True)
    plt.show()

# 生成 DAG 任务依赖图
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

# 参数设置
num_tasks = 8
num_nodes = 3
structure = "FFT"
if structure == "FFT":
    E = generate_fft_dag(num_tasks)
else:
    raise ValueError("Unsupported structure")

t_vm = np.random.randint(2, 10, size=(num_tasks, num_nodes))  # 执行时间
c_mm = np.random.randint(1, 5, size=(num_nodes, num_nodes))  # 节点通信延迟
np.fill_diagonal(c_mm, 0)
C_m = np.random.randint(10, 20, size=num_nodes)  # 节点显存容量
r_vm = np.random.randint(1, 5, size=(num_tasks, num_nodes))  # 任务显存需求
C_min = 2  # 任务最小显存需求

# Gurobi 模型
model = gp.Model("task_scheduling")

# 决策变量
z = model.addVars(num_tasks, num_nodes, vtype=GRB.BINARY, name="z")
T_v = model.addVars(num_tasks, vtype=GRB.CONTINUOUS, name="T_v")
T_makespan = model.addVar(vtype=GRB.CONTINUOUS, name="T_makespan")
w = model.addVars(num_tasks, num_tasks, num_nodes, num_nodes, vtype=GRB.BINARY, name="w")

# 目标函数
model.setObjective(T_makespan, GRB.MINIMIZE)

# 约束1：任务唯一分配约束
for v in range(num_tasks):
    model.addConstr(gp.quicksum(z[v, m] for m in range(num_nodes)) == 1)

# 约束2：节点计算资源容量约束
for m in range(num_nodes):
    model.addConstr(gp.quicksum(z[v, m] * r_vm[v, m] for v in range(num_tasks)) <= C_m[m])

# 约束3：任务最小显存需求
for v in range(num_tasks):
    for m in range(num_nodes):
        model.addConstr(z[v, m] * C_min <= C_m[m])

# 约束4：任务依赖约束（线性化）
for (v, v_prime) in E:
    for m in range(num_nodes):
        for m_prime in range(num_nodes):
            model.addConstr(w[v, v_prime, m, m_prime] <= z[v, m])
            model.addConstr(w[v, v_prime, m, m_prime] <= z[v_prime, m_prime])
            model.addConstr(w[v, v_prime, m, m_prime] >= z[v, m] + z[v_prime, m_prime] - 1)

    # 任务完成时间约束（包括通信时延）
    model.addConstr(T_v[v_prime] >= T_v[v] +
                    gp.quicksum(w[v, v_prime, m, m_prime] * (t_vm[v, m] + c_mm[m, m_prime])
                                for m in range(num_nodes) for m_prime in range(num_nodes)))

# 约束5：Makespan 定义约束
for v in range(num_tasks):
    model.addConstr(T_makespan >= T_v[v] + gp.quicksum(z[v, m] * t_vm[v, m] for m in range(num_nodes)))

# 约束6：非负时间约束
for v in range(num_tasks):
    model.addConstr(T_v[v] >= 0)

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal Makespan: {T_makespan.X:.2f}")
    print(f"Optomal Power: {GRB}" )
    t_finish = [T_v[v].X for v in range(num_tasks)]
    z_values = np.array([[z[v, m].X for m in range(num_nodes)] for v in range(num_tasks)])
    print("Task Completion Times:", t_finish)
    print("Task Allocation Matrix (z):")
    print(z_values)
    plot_gantt_chart(t_finish, t_vm, z_values, num_nodes)
else:
    print("No optimal solution found.")

# 修复后的优化模型
import cvxpy as cp
import numpy as np
import random
import matplotlib.pyplot as plt 
from gurobipy import Model, GRB, quicksum
import numpy as np
import seaborn as sns
# 固定随机种子，以保证结果的可重复性
random.seed(42)
np.random.seed(42)

# 绘图函数
def plot_makespan_vs_num_tasks(num_tasks, makespan, title=None):
    """绘制任务数量与工期之间的关系"""
    sns.set(style="white")  # 设置无网格的白色背景

    # 定义颜色和样式
    color = 'dodgerblue'  # 选择一种颜色
    marker = 'o'  # 选择标记样式

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制主曲线
    ax.plot(num_tasks, makespan, 
            marker=marker,  # 使用标记
            color=color,
            linestyle='-',  
            linewidth=2,  # 增加线条宽度
            markersize=8)   # 增大标记大小

    # 设置轴和标题
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Number of Tasks', size=14)
    ax.set_ylabel('Makespan', size=14)
    if title:
        ax.set_title(title, size=16)

    # 调整间隙
    ax.set_xticks(np.arange(min(num_tasks), max(num_tasks) + 1, 1))  # 设置x轴间隔
    ax.set_xticklabels(np.arange(min(num_tasks), max(num_tasks) + 1, 1))

    # 保存和显示
    fig.tight_layout()
    plt.show()
    return
    
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

    for v in range(num_tasks):
        for m in range(num_nodes):
            if np.all(r_vm[v, m] <= node_resources[m]):
                z[v, m] = 1
                node_resources[m] -= r_vm[v, m]

                # 确定任务开始时间，考虑依赖关系
                predecessors = [t_start[u] + t_vm[u, m] for u, v_prime in E if v_prime == v]
                t_start[v] = max(predecessors) if predecessors else 0  # 处理空序列
                break

    T_greedy = max(t_start[v] + t_vm[v, m] for v in range(num_tasks) for m in range(num_nodes) if z[v, m] > 0.5)
    return T_greedy, t_start, z

# 随机调度算法
def random_scheduling(t_vm, num_nodes, E):
    num_tasks = t_vm.shape[0]
    z = np.zeros((num_tasks, num_nodes))
    
    # 记录每个任务的开始时间
    t_start = np.zeros(num_tasks)
    t_end = np.zeros(num_tasks)  # 结束时间

    for v in range(num_tasks):
        # 随机选择一个节点进行任务分配
        selected_node = random.randint(0, num_nodes - 1)
        z[v, selected_node] = 1  # 将任务分配给选定的节点

        # 确定任务开始时间，考虑依赖关系
        predecessors = [t_end[u] for u, v_prime in E if v_prime == v]  # 获取所有前置任务的结束时间
        t_start[v] = max(predecessors) if predecessors else 0  # 如果没有前置任务，则开始时间为0
        t_end[v] = t_start[v] + t_vm[v, selected_node]  # 计算结束时间

    makespan = max(t_end)  # 工期是所有任务的最大结束时间
    return makespan, z

# 启发式调度算法（模拟退火算法实现）
def heuristic_scheduling(t_vm, c_mm, r_vm, C_m, E, initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
    num_tasks, num_nodes = t_vm.shape
    z = np.zeros((num_tasks, num_nodes))  # 任务分配
    best_solution = None
    best_makespan = float('inf')

    # 初始化随机解
    current_solution = np.zeros((num_tasks, num_nodes))
    node_resources = C_m.copy()  # 每个节点的剩余资源
    t_start = np.zeros(num_tasks)  # 每个任务的开始时间
    t_end = np.zeros(num_tasks)  # 每个任务的结束时间

    for v in range(num_tasks):
        selected_node = random.randint(0, num_nodes - 1)
        current_solution[v, selected_node] = 1
        node_resources[selected_node] -= r_vm[v, selected_node]

    # 计算当前解的工期
    for v in range(num_tasks):
        # 确定任务开始时间，考虑依赖关系
        predecessors = [t_end[u] for u, v_prime in E if v_prime == v]  # 获取所有前置任务的结束时间
        t_start[v] = max(predecessors) if predecessors else 0  # 如果没有前置任务，则开始时间为0
        t_end[v] = t_start[v] + t_vm[v, np.argmax(current_solution[v])]  # 计算结束时间

    current_makespan = max(t_end)

    temperature = initial_temp

    for iteration in range(max_iterations):
        # 生成新解
        new_solution = current_solution.copy()
        v = random.randint(0, num_tasks - 1)
        selected_node = random.randint(0, num_nodes - 1)

        # 尝试改变任务分配
        new_solution[v] = np.zeros(num_nodes)
        new_solution[v, selected_node] = 1

        # 计算新解的工期
        for v in range(num_tasks):
            # 确定任务开始时间，考虑依赖关系
            predecessors = [t_end[u] for u, v_prime in E if v_prime == v]  # 获取所有前置任务的结束时间
            t_start[v] = max(predecessors) if predecessors else 0  # 如果没有前置任务，则开始时间为0
            t_end[v] = t_start[v] + t_vm[v, np.argmax(new_solution[v])]  # 计算结束时间

        new_makespan = max(t_end)

        # 计算能量差
        delta_makespan = new_makespan - current_makespan

        # 根据能量差和温度决定是否接受新解
        if delta_makespan < 0 or random.uniform(0, 1) < np.exp(-delta_makespan / temperature):
            current_solution = new_solution
            current_makespan = new_makespan

            # 更新最佳解
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = current_solution

        # 降温
        temperature *= cooling_rate

    return best_makespan, np.zeros(num_tasks), best_solution  # 返回最佳工期和分配

# 启发式调度算法（GRASP实现）
def grasp_scheduling(t_vm, c_mm, r_vm, C_m, E, num_iterations=100, alpha=0.3):
    num_tasks, num_nodes = t_vm.shape
    best_solution = None
    best_makespan = float('inf')

    for _ in range(num_iterations):
        # 随机化贪婪构造阶段
        current_solution = np.zeros((num_tasks, num_nodes))  # 当前任务分配
        node_resources = C_m.copy()  # 每个节点的剩余资源
        t_start = np.zeros(num_tasks)  # 每个任务的开始时间

        # 选择候选节点
        for v in range(num_tasks):
            candidates = []
            for m in range(num_nodes):
                if np.all(r_vm[v, m] <= node_resources[m]):  # 检查资源是否足够
                    candidates.append(m)

            # 随机选择一部分候选节点
            if candidates:
                num_candidates = max(1, int(len(candidates) * alpha))  # 选择一定比例的候选节点
                selected_node = random.choice(candidates[:num_candidates])  # 从候选中随机选择一个节点

                current_solution[v, selected_node] = 1
                node_resources[selected_node] -= r_vm[v, selected_node]

                # 确定任务开始时间，考虑依赖关系
                predecessors = [t_start[u] + t_vm[u, selected_node] for u, v_prime in E if v_prime == v]
                t_start[v] = max(predecessors) if predecessors else 0  # 处理空序列

        # 计算当前解的工期
        current_makespan = max(t_start[v] + t_vm[v, m] for v in range(num_tasks) for m in range(num_nodes) if current_solution[v, m] > 0.5)

        # 局部搜索阶段
        for v in range(num_tasks):
            for m in range(num_nodes):
                if np.all(r_vm[v, m] <= C_m[m]):  # 检查资源是否足够
                    # 尝试将任务 v 分配给节点 m
                    temp_solution = current_solution.copy()
                    temp_solution[v] = np.zeros(num_nodes)
                    temp_solution[v, m] = 1

                    # 计算新解的工期
                    temp_makespan = max(t_start[v] + t_vm[v, m] for v in range(num_tasks) for m in range(num_nodes) if temp_solution[v, m] > 0.5)

                    # 更新当前解
                    if temp_makespan < current_makespan:
                        current_solution = temp_solution
                        current_makespan = temp_makespan

        # 更新最佳解
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_solution = current_solution

    return best_makespan, np.zeros(num_tasks), best_solution  # 返回最佳工期和分配

    
# 生成 DAG 结构的函数
def generate_ge_dag(num_tasks):
    edges = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if random.random() < 0.5:  # 随机决定是否添加依赖关系
                edges.append((i, j))
    return edges

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

def generate_ge_dag_new(dim):
    """
    Generate a Directed Acyclic Graph (DAG) using the Gaussian Elimination (GE) method.

    Parameters:
        dim (int): The dimension \(\eta\) of the DAG.

    Returns:
        G (networkx.DiGraph): The generated DAG.
    """
    # Number of tasks based on the formula
    num_tasks = (dim**2 + dim - 2) // 2

    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(num_tasks):
        G.add_node(i, label=f"Task {i+1}")

    # Add edges based on Gaussian Elimination structure
    for i in range(dim):
        for j in range(i + 1, dim):
            task_from = i * dim + j - (i * (i + 1)) // 2 - 1
            task_to = j * dim + j - (j * (j + 1)) // 2 - 1
            if task_from < num_tasks and task_to < num_tasks:
                G.add_edge(task_from, task_to)

    return G

def generate_fft_dag_new(theta):
    """
    Generate a Directed Acyclic Graph (DAG) using the Fast Fourier Transform (FFT) method.

    Parameters:
        theta (int): The number of FFT points \(\theta\).

    Returns:
        G (networkx.DiGraph): The generated DAG.
    """
    # Number of recursive call tasks and butterfly tasks
    num_recursive_tasks = 2 * (theta - 1) + 1
    num_butterfly_tasks = int(theta * math.log2(theta))

    # Total number of tasks
    num_tasks = num_recursive_tasks + num_butterfly_tasks

    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(num_tasks):
        G.add_node(i, label=f"Task {i+1}")

    # Add edges for recursive calls
    for i in range(1, num_recursive_tasks):
        G.add_edge(i - 1, i)  # Sequential dependencies for recursive calls

    # Add edges for butterfly operations
    start_butterfly = num_recursive_tasks
    for i in range(start_butterfly, num_tasks):
        src_task = (i - start_butterfly) // 2
        G.add_edge(src_task, i)  # Connect recursive calls to butterfly tasks

    return G


# 选择 DAG 结构的函数
def generate_dag(num_tasks, structure):
    if structure == "GE":
        return generate_ge_dag(num_tasks)
    elif structure == "FFT":
        return generate_fft_dag(num_tasks)
    else:
        raise ValueError("Unsupported DAG structure")

# 计算给定 DAG 的 makespan
def calculate_makespan(E, t_vm, z):
    num_tasks, num_nodes = t_vm.shape
    t_start = np.zeros(num_tasks)
    
    for v in range(num_tasks):
        for m in range(num_nodes):
            if z[v, m] > 0.5:  # 任务分配到节点 m
                t_start[v] = max(t_start[v], t_start[v] + t_vm[v, m])
                break

    return max(t_start)

# 启发式凸优化算法
def heuristic_convex_optimization_scheduling(structure, num_tasks, num_nodes, B_total, c_mm):
    # 生成 DAG 结构
    E = generate_dag(num_tasks, structure)

    # 资源和时间矩阵的初始化
    t_vm = np.random.rand(num_tasks, num_nodes)  # 假设的任务-节点时间矩阵
    r_vm = np.random.rand(num_tasks, num_nodes)  # 假设的任务-节点资源矩阵
    C_m = np.random.rand(num_nodes) * B_total  # 节点的总资源

    # 使用启发式方法生成初始解
    initial_makespan, _, initial_solution = heuristic_scheduling(t_vm, c_mm, r_vm, C_m, E)

    # 使用凸优化算法进一步优化
    T_makespan, best_solution = convex_optimization_scheduling(E, t_vm, r_vm, C_m, c_mm, num_tasks, num_nodes)

    # 如果启发式方法的结果更好，则更新 T_makespan
    if initial_makespan < T_makespan:
        T_makespan = initial_makespan
        best_solution = initial_solution

    # 返回最终的工期和分配
    return T_makespan, best_solution

# 凸优化调度算法
def convex_optimization_scheduling(E, t_vm, r_vm, C_m, c_mm, num_tasks, num_nodes):
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
    
    return T.value, np.round(z.value, decimals=2)

def DGF_algorithm(structure='FFT', num_tasks=30, num_nodes=5,B_total=[50],P_max=[60], selected_algorithm="greedy"):
    # 主程序 #修改任务数量，节点数量和 能耗
    # structure = "FFT"  # 可选 "GE" 或 "FFT"
    # num_tasks  # 任务数量
    # num_nodes = 5  # 节点数量
    # B_total = [50]  # Total bandwidth
    # P_max = [60]  # Maximum power
    # # 选择算法
    # selected_algorithm = "greedy"  # 可选 "random", "greedy", "heuristic", "convex"

    # 生成 DAG
    E = generate_dag(num_tasks, structure)
    print("生成的任务依赖关系:", E)

    # 随机生成执行时间和资源需求
    t_vm = np.random.randint(2, 10, size=(num_tasks, num_nodes))
    c_mm = np.random.randint(1, 5, size=(num_nodes, num_nodes))
    np.fill_diagonal(c_mm, 0)  # 自身通信延迟为 0
    C_m = np.random.randint(10, 20, size=num_nodes)  # 随机生成每个节点的资源容量
    r_vm = np.random.randint(1, 5, size=(num_tasks, num_nodes))  # 随机生成 1 到 5 的资源需求

    # 根据选择的算法进行调度
    if selected_algorithm == "convex":
        T_makespan, Task_allocation = convex_optimization_scheduling(E, t_vm, r_vm, C_m, c_mm, num_tasks, num_nodes)
        print("总工期 T:", T_makespan, Task_allocation)
        # plot_gantt_chart(np.zeros(num_tasks), t_vm, Task_allocation, num_nodes)  # 这里假设开始时间为0
    elif selected_algorithm == "greedy":
        T_makespan, greedy_t_start, Task_allocation = greedy_scheduling(t_vm, c_mm, r_vm, C_m, E)
        print("贪婪算法总工期:", T_makespan)
    elif selected_algorithm == "heuristic":
        T_makespan, heuristic_t_start, Task_allocation = heuristic_scheduling(t_vm, c_mm, r_vm, C_m, E)
        print("启发式算法总工期:", T_makespan, Task_allocation)
    elif selected_algorithm == "random":
        T_makespan, Task_allocation = random_scheduling(t_vm, num_nodes,E)
        print("随机调度结果:", T_makespan, Task_allocation)
    elif selected_algorithm == "grasp":
        T_makespan, grasp_t_start, Task_allocation = grasp_scheduling(t_vm, c_mm, r_vm, C_m, E, num_iterations=100, alpha=0.3)
        print("GRASP算法总工期:", T_makespan, Task_allocation)
    elif selected_algorithm == 'heuristic_convex':
        T_makespan, Task_allocation = heuristic_convex_optimization_scheduling(structure, num_tasks, num_nodes, B_total, c_mm)
        print("heuristic_convex算法总工期:", T_makespan, Task_allocation)
    else:
        print("无效的选择，请选择有效的算法。")

    print('------------------------phase 1 finished -----------------------------------')





    # Convert makespan to time slots
    Delta_t = 1  # Slot duration
    T_debug = int(T_makespan // Delta_t)

    # Define communication links based on Task_allocation
    L_debug = [
        (i, j)
        for i in range(Task_allocation.shape[0])
        for j in range(Task_allocation.shape[1])
        if Task_allocation[i, j] == 1
    ]

    # Define data demands for each link based on tasks
    O_v_updated = {link: 5 for link in L_debug}  # Example data demands

    # Update model parameters

    phi_k_buffered = np.linspace(0.28, 1.26, 3 + 1)  # Adjusted for reduced segments
    p_sm_lower_bound = 0.1  # Minimum power

    # Create the updated model
    model_task_allocation = Model("PLA_Task_Allocation_Model")

    # Decision variables
    p_sm_task = model_task_allocation.addVars(len(L_debug), T_debug, vtype=GRB.CONTINUOUS, lb=p_sm_lower_bound, name="p_sm")
    delta_k_task = model_task_allocation.addVars(len(L_debug), T_debug, 3, vtype=GRB.BINARY, name="delta_k")
    slack_phi_lower_task = model_task_allocation.addVars(len(L_debug), T_debug, 3, vtype=GRB.CONTINUOUS, name="slack_phi_lower")
    slack_phi_upper_task = model_task_allocation.addVars(len(L_debug), T_debug, 3, vtype=GRB.CONTINUOUS, name="slack_phi_upper")

    # Objective: Minimize quadratic power and penalize slack variables
    model_task_allocation.setObjective(
        quicksum(p_sm_task[l, t] ** 2 for l in range(len(L_debug)) for t in range(T_debug)) +
        10 * quicksum(slack_phi_lower_task[l, t, k] + slack_phi_upper_task[l, t, k] for l in range(len(L_debug)) for t in range(T_debug) for k in range(3)),
        GRB.MINIMIZE
    )

    # Constraints
    # Data transmission requirement
    for l, link in enumerate(L_debug):
        model_task_allocation.addConstr(
            quicksum(B_total[0] * delta_k_task[l, t, k] for t in range(T_debug) for k in range(3)) >= O_v_updated[link],
            name=f"data_transmission_{l}"
        )

    # Bandwidth allocation constraint
    for t in range(T_debug):
        model_task_allocation.addConstr(
            quicksum(delta_k_task[l, t, k] for l in range(len(L_debug)) for k in range(3)) <= B_total[0] * 0.7,
            name=f"bandwidth_allocation_t{t}"
        )

    # Power constraints
    for l in range(len(L_debug)):
        for t in range(T_debug):
            model_task_allocation.addConstr(
                p_sm_task[l, t] <= P_max[0] * 0.9,
                name=f"power_constraint_{l}_t{t}"
            )
            model_task_allocation.addConstr(
                p_sm_task[l, t] >= p_sm_lower_bound,
                name=f"power_lower_bound_{l}_t{t}"
            )

    # PLA activation logic constraints with slack variables
    for l in range(len(L_debug)):
        for t in range(T_debug):
            phi_sm = p_sm_task[l, t] * 1 / (1 + 0.1)  # Using example gain and interference
            model_task_allocation.addConstr(
                quicksum(delta_k_task[l, t, k] for k in range(3)) == 1,
                name=f"active_segment_{l}_t{t}"
            )
            for k in range(3):
                model_task_allocation.addConstr(
                    phi_k_buffered[k] * delta_k_task[l, t, k] - slack_phi_lower_task[l, t, k] <= phi_sm,
                    name=f"phi_lower_{l}_t{t}_k{k}"
                )
                model_task_allocation.addConstr(
                    phi_sm <= phi_k_buffered[k + 1] * delta_k_task[l, t, k] + slack_phi_upper_task[l, t, k],
                    name=f"phi_upper_{l}_t{t}_k{k}"
                )

    # Solve the updated model
    model_task_allocation.optimize()

    # Extract the optimal transmission power if feasible
    solution_task_allocation_info = {}
    if model_task_allocation.status == GRB.OPTIMAL:
        solution_p_sm_task = model_task_allocation.getAttr("x", p_sm_task)
        E_comm_task = sum(solution_p_sm_task[(l, t)] * Delta_t for l in range(len(L_debug)) for t in range(T_debug))
        solution_task_allocation_info = {
            "Communication Energy (E_comm)": E_comm_task,
            "Optimal Power (p_sm)": solution_p_sm_task
        }
    else:
        solution_task_allocation_info = {
            "Status": model_task_allocation.status
        }
    best_objective = model_task_allocation.objVal
    # solution_task_allocation_info
    return T_makespan, best_objective

if __name__ =='__main__':
    structure = "FFT"  # 可选 "GE" 或 "FFT"
    num_tasks= 35  # 任务数量
    num_nodes = 5  # 节点数量
    B_total = [50]  # Total bandwidth
    P_max = [60]  # Maximum power
    # 选择算法
    selected_algorithm = "convex"  # 可选 "random", "greedy", "heuristic", "convex"，'grasp'
    DGF_algorithm(structure,num_tasks,num_nodes,B_total,P_max,selected_algorithm)

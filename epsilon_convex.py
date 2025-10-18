import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子
np.random.seed(42)

# 绘制 Pareto 前沿
def plot_pareto_front(results):
    if len(results) == 0:
        print("No feasible solutions found for the given epsilon values.")
        return

    results = np.array(results)
    plt.figure(figsize=(8, 6))
    plt.plot(results[:, 0], results[:, 1], marker='o', label="Pareto Front")
    plt.xlabel("Makespan (T_makespan)")
    plt.ylabel("Total Energy Consumption (E_total)")
    plt.title("Pareto Front")
    plt.grid(True)
    plt.legend()
    plt.show()

# 调度函数
def task_scheduling(method="weighted", alpha=0.5, epsilon_values=None):
    # 参数设置
    num_tasks = 8
    num_nodes = 4  # 节点包括无人船、无人机、海事船、海洋基站

    # 节点参数（算力上限，单位：GFLOPs）
    C_m = np.array([100, 50, 500, 2000])
    # 显存容量（单位：GB）
    M_m = np.array([16, 8, 64, 256])

    # 能耗参数
    kappa_e = np.array([0.02, 0.01, 0.05, 0.1])
    beta_e = np.array([0.03, 0.02, 0.04, 0.05])
    gamma_e = np.array([0.03, 0.02, 0.04, 0.05])

    # 通信延迟矩阵 (单位：ms)
    c_mm = np.array([
        [1e-6, 100, 200, 100],   # 无人船到其他
        [100, 1e-6, 200, 20],    # 无人机到其他
        [200, 200, 1e-6, 50],    # 海事船到其他
        [100, 20, 50, 1e-6]      # 海洋基站到其他
    ])

    # 任务参数
    t_vm = np.random.randint(5, 51, size=(num_tasks, len(C_m)))  # 推理需求 GFLOPs
    r_vm = np.random.randint(1, 5, size=(num_tasks, len(C_m)))  # 最小显存需求 GB
    I_v = np.random.uniform(0.1, 5, size=num_tasks)  # 输入数据 MB
    O_v = np.random.uniform(0.1, 5, size=num_tasks)  # 输出数据 MB

    if method == "weighted":
        model = gp.Model("task_scheduling_weighted")

        # 决策变量
        z = model.addVars(num_tasks, len(C_m), vtype=GRB.BINARY, name="z")
        T_v = model.addVars(num_tasks, vtype=GRB.CONTINUOUS, name="T_v")
        T_makespan = model.addVar(vtype=GRB.CONTINUOUS, name="T_makespan")

        # 目标函数
        E_total = gp.quicksum(z[v, m] * (
            kappa_e[m] * t_vm[v, m] +
            beta_e[m] * I_v[v] / max(c_mm[m, m], 1e-6) +
            gamma_e[m] * O_v[v] / max(c_mm[m, m], 1e-6)
        ) for v in range(num_tasks) for m in range(len(C_m)))

        model.setObjective(alpha * T_makespan + (1 - alpha) * E_total, GRB.MINIMIZE)

        # 约束1：任务唯一分配约束
        for v in range(num_tasks):
            model.addConstr(gp.quicksum(z[v, m] for m in range(len(C_m))) == 1)

        # 约束2：节点容量约束（算力）
        for m in range(len(C_m)):
            model.addConstr(gp.quicksum(z[v, m] * t_vm[v, m] for v in range(num_tasks)) <= C_m[m])

        # 约束3：节点容量约束（显存）
        for m in range(len(C_m)):
            model.addConstr(gp.quicksum(z[v, m] * r_vm[v, m] for v in range(num_tasks)) <= M_m[m])

        # 约束4：Makespan 定义约束
        for v in range(num_tasks):
            model.addConstr(T_makespan >= T_v[v] + gp.quicksum(z[v, m] * t_vm[v, m] for m in range(len(C_m))))

        # 约束5：非负时间约束
        for v in range(num_tasks):
            model.addConstr(T_v[v] >= 0)

        # 求解模型
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print(f"Optimal Makespan (T_makespan): {T_makespan.X:.2f}")
            print(f"Optimal Total Energy Consumption (E_total): {E_total.getValue():.2f}")

            # 提取变量值用于验证
            z_values = [[z[v, m].X for m in range(len(C_m))] for v in range(num_tasks)]
            T_v_values = [T_v[v].X for v in range(num_tasks)]
            return T_makespan.X, E_total.getValue(), z_values, T_v_values, t_vm, c_mm, r_vm
        else:
            print("No optimal solution found.")
            return None, None, None, None, None, None, None

    elif method == "epsilon":
        if epsilon_values is None:
            raise ValueError("epsilon_values must be provided for epsilon method")

        pareto_results = []
        for epsilon in epsilon_values:
            model = gp.Model(f"task_scheduling_epsilon_{epsilon}")

            # 决策变量
            z = model.addVars(num_tasks, len(C_m), vtype=GRB.BINARY, name="z")
            T_v = model.addVars(num_tasks, vtype=GRB.CONTINUOUS, name="T_v")
            T_makespan = model.addVar(vtype=GRB.CONTINUOUS, name="T_makespan")

            # 目标函数
            model.setObjective(T_makespan, GRB.MINIMIZE)

            # 约束1：任务唯一分配约束
            for v in range(num_tasks):
                model.addConstr(gp.quicksum(z[v, m] for m in range(len(C_m))) == 1)

            # 约束2：节点容量约束（算力）
            for m in range(len(C_m)):
                model.addConstr(gp.quicksum(z[v, m] * t_vm[v, m] for v in range(num_tasks)) <= C_m[m])

            # 约束3：节点容量约束（显存）
            for m in range(len(C_m)):
                model.addConstr(gp.quicksum(z[v, m] * r_vm[v, m] for v in range(num_tasks)) <= M_m[m])

            # 约束4：Makespan 定义约束
            for v in range(num_tasks):
                model.addConstr(T_makespan >= T_v[v] + gp.quicksum(z[v, m] * t_vm[v, m] for m in range(len(C_m))))

            # 约束5：非负时间约束
            for v in range(num_tasks):
                model.addConstr(T_v[v] >= 0)

            # ε-约束
            E_total = gp.quicksum(z[v, m] * (
                kappa_e[m] * t_vm[v, m] +
                beta_e[m] * I_v[v] / max(c_mm[m, m], 1e-6) +
                gamma_e[m] * O_v[v] / max(c_mm[m, m], 1e-6)
            ) for v in range(num_tasks) for m in range(len(C_m)))

            model.addConstr(E_total <= epsilon)

            # 求解模型
            model.optimize()

            if model.status == GRB.OPTIMAL:
                pareto_results.append((T_makespan.X, E_total.getValue()))

        # 绘制 Pareto 前沿
        plot_pareto_front(pareto_results)
        return pareto_results
    else:
        raise ValueError("Unsupported method. Use 'weighted' or 'epsilon'.")

def verify_results(t_vm, c_mm, r_vm, z_values, T_v_values, num_tasks, num_nodes, kappa_e, beta_e, gamma_e):
    # 手动计算 Makespan
    makespan_manual = max(
        T_v_values[v] + sum(z_values[v][m] * t_vm[v, m] for m in range(num_nodes))
        for v in range(num_tasks)
    )

    # 手动计算 Total Energy Consumption
    total_energy_manual = sum(
        z_values[v][m] * (
            kappa_e[m] * t_vm[v, m] +
            beta_e[m] * r_vm[v, m] / max(c_mm[m, m], 1e-6) +  # 输入通信能耗
            gamma_e[m] * O_v[v] / max(c_mm[m, m], 1e-6)   # 输出通信能耗
        )
        for v in range(num_tasks) for m in range(num_nodes)
    )

    print(f"Manual Makespan: {makespan_manual}")
    print(f"Manual Total Energy Consumption: {total_energy_manual}")

    return makespan_manual, total_energy_manual

if __name__ == "__main__":
    method = "epsilon"  # 或 "weighted"
    if method == "weighted":
        alpha = 0.5  # 加权和法的权重
        T_makespan, E_total, z_values, T_v_values, t_vm, c_mm, r_vm = task_scheduling(method="weighted", alpha=alpha)

        # 手动验证
        if T_makespan is not None and E_total is not None:
            makespan_manual, energy_manual = verify_results(
                t_vm, c_mm, r_vm, z_values, T_v_values, num_tasks=8, num_nodes=4,
                kappa_e=np.array([0.02, 0.01, 0.05, 0.1]), 
                beta_e=np.array([0.03, 0.02, 0.04, 0.05]),
                gamma_e=np.array([0.03, 0.02, 0.04, 0.05])
            )

            # 对比结果
            print(f"Gurobi Makespan: {T_makespan}, Manual Makespan: {makespan_manual}")
            print(f"Gurobi Total Energy: {E_total}, Manual Total Energy: {energy_manual}")
    elif method == "epsilon":
        epsilon_values = np.linspace(50, 200, 5)  # ε-约束法的 ε 值
        pareto_results = task_scheduling(method="epsilon", epsilon_values=epsilon_values)

        if pareto_results:
            for makespan, energy in pareto_results:
                print(f"Pareto Point - Makespan: {makespan}, Energy: {energy}")
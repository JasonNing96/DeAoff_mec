import cvxpy as cp
import numpy as np

# 参数定义
num_links = 10
T_max = 50
Delta_t = 1
B_total = 100
P_max = 10
N0 = 1e-9
I = np.random.rand(num_links, T_max)  # 假设干扰是已知的
g = np.random.rand(num_links, T_max)  # 信道增益

# 决策变量
p = cp.Variable((num_links, T_max), nonneg=True)  # 功率分配
b = cp.Variable((num_links, T_max), nonneg=True)  # 带宽分配
r_tilde = cp.Variable((num_links, T_max), nonneg=True)  # 分段线性化后的速率
z = cp.Variable((num_links, T_max), nonneg=True)  # 辅助变量 z = b * r_tilde

# 分段线性化参数
K = 4  # 分段数量
# phi_bounds = np.linspace(0, 10, K + 1)  # 分段边界
# phi_bounds = np.linspace(0, 50, K + 1)
phi_bounds = np.linspace(1e-3, 50, K + 1)  # 避免 log(0) 的情况
# alpha = np.random.rand(K)  # 线性近似的斜率
# beta = np.random.rand(K)  # 线性近似的截距
alpha = [(np.log2(1 + phi_bounds[k + 1]) - np.log2(1 + phi_bounds[k])) / (phi_bounds[k + 1] - phi_bounds[k]) for k in range(K)]
beta = [np.log2(1 + phi_bounds[k]) - alpha[k] * phi_bounds[k] for k in range(K)]
delta = [cp.Variable((num_links, K), boolean=True) for _ in range(T_max)]  # 每个时间槽的分段变量

phi = cp.multiply(p, g) / (N0 + I)  # SINR
M = 1e6  # 大M常数

# 目标函数：最小化通信能耗
objective = cp.Minimize(cp.sum(cp.multiply(p, Delta_t)))

# 约束条件
constraints = []

# 总带宽约束
for t in range(T_max):
    constraints.append(cp.sum(b[:, t]) <= B_total)

# 数据传输完成约束
# O_v = np.random.rand(num_links)  # 假设每个任务的数据量是已知的
# O_v = np.random.uniform(10, 100, num_links)  # 增大任务数据量
O_v = np.linspace(10, 500, num_links)

for i in range(num_links):
    constraints.append(cp.sum(z[i, :] * Delta_t) >= O_v[i])  # 数据传输完成

# 功率和带宽限制
constraints.append(p <= P_max)

constraints.append(b >= 1e-3)
constraints.append(p >= 1e-3)


# 分段线性化约束
for t in range(T_max):
    delta_t = delta[t]
    for link_idx in range(num_links):
        # 确保仅激活一个分段
        constraints.append(cp.sum(delta_t[link_idx, :]) == 1)

        # 替换分段约束
        for k in range(K):
            constraints.append(phi_bounds[k] * delta_t[link_idx, k] <= phi[link_idx, t])
            constraints.append(phi[link_idx, t] <= phi_bounds[k + 1] * delta_t[link_idx, k])
            constraints.append(r_tilde[link_idx, t] <= alpha[k] * phi[link_idx, t] + beta[k] + (1 - delta_t[link_idx, k]) * M)

# 线性化传输速率约束
for t in range(T_max):
    for link_idx in range(num_links):
        constraints.append(z[link_idx, t] <= M * b[link_idx, t])
        constraints.append(z[link_idx, t] <= M * r_tilde[link_idx, t])
        constraints.append(z[link_idx, t] >= b[link_idx, t] + r_tilde[link_idx, t] - M)
        # constraints.append(z[link_idx, t] >= b[link_idx, t] * r_tilde[link_idx, t] - (1 - delta_t[link_idx, k]) * M)



# 求解问题
problem = cp.Problem(objective, constraints)
print("Number of variables:", problem.size_metrics.num_scalar_variables)
print("Number of constraints:", problem.size_metrics.num_scalar_eq_constr + problem.size_metrics.num_scalar_leq_constr)

try:
    problem.solve(solver=cp.GUROBI, verbose=True)
    if problem.status == cp.OPTIMAL:
        print("Optimal Solution Found!")
        p_star = p.value
        b_star = b.value
        r_tilde_star = r_tilde.value
        z_star = z.value
        print("Power allocation:", p_star)
        print("Bandwidth allocation:", b_star)
        print("Rate (tilde):", r_tilde_star)
        print("Auxiliary variable z:", z_star)
    else:
        print("Problem status:", problem.status)
except Exception as e:
    print("Solver error:", e)

from gurobipy import Model, GRB, quicksum
import numpy as np
def validate(p_val, b_val, r_tilde_val, z_val, O_v, B_total, P_max, Delta_t):
    """
    验证优化结果是否满足约束
    """
    valid = True
    messages = []

    # 验证功率约束
    if not (np.all(p_val >= 0) and np.all(p_val <= P_max)):
        messages.append("功率约束未满足")
        valid = False

    # 验证带宽总量约束
    if not np.all(np.sum(b_val, axis=0) <= B_total):
        messages.append("带宽总量约束未满足")
        valid = False

    # 验证数据需求
    data_transmitted = np.sum(z_val * Delta_t, axis=1)
    if not np.all(data_transmitted >= O_v):
        messages.append("数据需求未满足")
        valid = False

    # 打印验证结果
    if valid:
        print("所有约束均已满足")
    else:
        print("验证失败，未满足的约束:")
        for msg in messages:
            print(f"- {msg}")



# Parameters
T_debug = 3  # Number of time slots
K_reduced = 7  # Number of PLA segments
p_sm_lower_bound = 0.1  # Minimum power
phi_k_buffered = [0.28, 0.42, 0.56, 0.7, 0.84, 0.98, 1.12, 1.26]  # Buffered phi_k ranges
B_total = [10]  # Total bandwidth
P_max = [1]  # Maximum power
g_sm = [[1] * T_debug]  # Channel gain (example values)
N_0 = 1  # Noise power
I_sm = [[0.1] * T_debug]  # Interference power

# Create the model
model = Model("PLA_Full_Constraints_Test")

# Decision variables
p_sm = model.addVars(1, T_debug, vtype=GRB.CONTINUOUS, lb=p_sm_lower_bound, name="p_sm")  # Transmission power
delta_k = model.addVars(1, T_debug, K_reduced, vtype=GRB.BINARY, name="delta_k")  # PLA binary vars
slack_phi_lower = model.addVars(1, T_debug, K_reduced, vtype=GRB.CONTINUOUS, name="slack_phi_lower")  # Slack for lower bounds
slack_phi_upper = model.addVars(1, T_debug, K_reduced, vtype=GRB.CONTINUOUS, name="slack_phi_upper")  # Slack for upper bounds

# Objective: Minimize quadratic power and penalize slack variables
model.setObjective(
    quicksum(p_sm[0, t] ** 2 for t in range(T_debug)) +
    quicksum(slack_phi_lower[0, t, k] + slack_phi_upper[0, t, k] for t in range(T_debug) for k in range(K_reduced)),
    GRB.MINIMIZE
)

# Constraints
# Data transmission requirement
model.addConstr(
    quicksum(B_total[0] * delta_k[0, t, k] for t in range(T_debug) for k in range(K_reduced)) >= 30,
    name="data_transmission"
)

# Bandwidth allocation constraint
for t in range(T_debug):
    model.addConstr(
        quicksum(delta_k[0, t, k] for k in range(K_reduced)) <= B_total[0] * 0.7,
        name=f"bandwidth_allocation_t{t}"
    )

# Power constraints
for t in range(T_debug):
    model.addConstr(
        p_sm[0, t] <= P_max[0] * 0.9,
        name=f"power_constraint_t{t}"
    )

# PLA activation logic constraints with slack variables
for t in range(T_debug):
    phi_sm = p_sm[0, t] * g_sm[0][t] / (N_0 + I_sm[0][t])  # Non-linear term
    # Ensure only one segment is active
    model.addConstr(
        quicksum(delta_k[0, t, k] for k in range(K_reduced)) == 1,
        name=f"active_segment_0_t{t}"
    )
    for k in range(K_reduced):
        # Segment constraints for phi with slack variables
        model.addConstr(
            phi_k_buffered[k] * delta_k[0, t, k] - slack_phi_lower[0, t, k] <= phi_sm,
            name=f"phi_lower_0_t{t}_k{k}"
        )
        model.addConstr(
            phi_sm <= phi_k_buffered[k + 1] * delta_k[0, t, k] + slack_phi_upper[0, t, k],
            name=f"phi_upper_0_t{t}_k{k}"
        )

# Solve the model
model.optimize()

# Extract and display results
if model.status == GRB.OPTIMAL:
    solution_p_sm = model.getAttr("x", p_sm)
    solution_delta_k = model.getAttr("x", delta_k)
    solution_slack_phi_lower = model.getAttr("x", slack_phi_lower)
    solution_slack_phi_upper = model.getAttr("x", slack_phi_upper)

    print("Optimal Power (p_sm):", solution_p_sm)
    # print("PLA Indicator (delta_k):", solution_delta_k)
    # print("Slack Lower:", solution_slack_phi_lower)
    # print("Slack Upper:", solution_slack_phi_upper)
else:
    print(f"Model status: {model.status}")

# if problem.status == cp.OPTIMAL:
#     print("Optimal Solution Found!")
#     validate(p.value, b.value, r_tilde.value, z.value, O_v, B_total, P_max, Delta_t)
#     print("Power allocation:\n", p.value)
#     print("Bandwidth allocation:\n", b.value)
# else:
#     print("Problem status:", problem.status)


#Parameters for energy calculation
Delta_t = 1  # Slot duration (adjust as needed)

# Extract the optimal transmission power
if model.status == GRB.OPTIMAL:
    solution_p_sm = model.getAttr("x", p_sm)

    # Calculate communication energy
    E_comm = sum(solution_p_sm[(0, t)] * Delta_t for t in range(T_debug))

    # For simplicity, assume no other energy component
    E_total = E_comm

    print(f"Communication Energy (E_comm): {E_comm}")
    print(f"Total Energy (E_total): {E_total}")
else:
    print(f"Model status: {model.status}")
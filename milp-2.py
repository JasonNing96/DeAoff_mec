from gurobipy import Model, GRB, quicksum
import numpy as np

# First phase output
T_makespan = 29
# Task_allocation = np.array([
#     [1., 0., 0., 0., 0.],
#     [1., 0., 0., 0., 0.],
#     [0., 0., 1., 0., 0.],
#     [0., 1., 0., 0., 0.],
#     [0., 1., 0., 0., 0.],
#     [0., 1., 0., 0., 0.],
#     [1., 0., 0., 0., 0.],
# ])
Task_allocation = np.array(
    [[0., 0., 1.],  # 添加逗号
    [1., 0., 0.],  # 添加逗号
    [1., 0., 0.],  # 添加逗号
    [0., 0., 1.],  # 添加逗号
    [0., 1., 0.],  # 添加逗号
    [1., 0., 0.],  # 添加逗号
    [1., 0., 0.],  # 添加逗号
    [1., 0., 0.]]  # 添加逗号
)

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
B_total = [100]  # Total bandwidth
P_max = [10]  # Maximum power
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

solution_task_allocation_info

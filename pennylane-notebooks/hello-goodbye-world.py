import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)

# Initial circuit, two rotations in one qubit
@qml.qnode(dev)
def circuit(params: list):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


params = [0, 0.5]
print(circuit(params))

# Cost function
def cost(x):
    return circuit(x)


init_params = np.array([0.11, 0.5])
print(cost(init_params))

# Optimize the function using Gradient Descent
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100
params = init_params

for i in range(steps):
    # updates the circuit params in each step
    params = opt.step(cost, params)
    if (i + 1) % 5 == 0:
        print(f"cost after step {i+1}: {cost(params)}")
print(f"Optimized rotation angles: {params}")

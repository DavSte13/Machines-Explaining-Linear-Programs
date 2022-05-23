import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
Generation of a larger random example for the maximum flow problem as well as the knapsack problem.
The random generated examples are not automatically stored.
"""
# generate a random instance of the knapsack problem
np.random.seed(42)

items = np.random.rand(50, 2)
print("np.array([")

for i in items:
    print(f"[{i[0]}, {i[1]}],")

print("])")

base1 = items/100

print("np.array([")

for i in range(len(base1)):
    if i%2 == 0:
        print(f"[{base1[i][0]}, {base1[i][1]}], [{base1[i + 1][0]}, {base1[i + 1][1]}],")

print("])")

average_value = np.average(items[:, 0])/10
average_volume = np.average(items[:, 1])/10
print(average_value, average_volume)


# generate a random graph for the maximum flow problem
# setup a random graph
n = 20
# generate the random graph and assert it has at least one path between source and sink
while True:
    G = nx.generators.random_graphs.erdos_renyi_graph(n, 0.3, seed=42, directed=True)
    if nx.has_path(G, 0, 19):
        break

colors = ['tab:olive'] + ['tab:blue']*18 + ['tab:orange']
# draw the graph
nx.draw_networkx(G, pos=nx.spring_layout(G), node_color=colors, with_labels=False, width=0.3, arrowsize=7)
#plt.show()
plt.savefig("graph_layout.pdf")

# add random edge weights
for (u, v, w) in G.edges(data=True):
    w['w'] = np.round(np.random.rand(), 3)

print(G.edges(data=True))

# base1 edges (near-zero)
for(u, v, w) in G.edges(data=True):
    w['w'] = w['w']/100

print(G.edges(data=True))

# base2 (all same base)
for(u, v, w) in G.edges(data=True):
    w['w'] = 0.001

print(G.edges(data=True))

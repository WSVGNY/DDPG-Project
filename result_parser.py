import matplotlib.pyplot as plt
import os
import csv 
from statistics import mean 

systems = ["arch", "manjaro", "ubuntu", "windows"]
results = {}
for s in systems:
    results[s] = {}
    results[s]["path"] = "results/results_" + s
    (_, _, filenames) = next(os.walk(results[s]["path"]))

    for name in filenames:
        if name[-3:] != "csv":
            continue

        split = name.split("_")

        codebase = split[0]
        lr = split[1].split("-")[1]

        if codebase not in results[s]:
            results[s][codebase] = {}
        results[s][codebase][lr] = []
        with open(results[s]["path"] + "/" + name, "r") as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                results[s][codebase][lr].append(row)

def showComparisionGraphs(x_index, y_index, codebase, lr):
    for s in systems:
        x = []
        y = []
        if lr in results[s][codebase]:
            for ep in results[s][codebase][lr]:
                if x_index == 0:
                    x.append(int( ep[x_index]))
                elif x_index == 1:
                    x.append(float( ep[x_index]))

                y.append(float(ep[y_index]))
            plt.plot(x, y, label=s)


    if x_index == 0:
        plt.xlabel("episode")
    elif x_index == 1:
        plt.xlabel("timestamp")

    plt.ylabel("score")

    if y_index == 2:
        plt.title("Score per episode")
    elif y_index == 3:
        plt.title("Average over last 100 episodes for learning rate " + lr)

    plt.ylim(-850, 250)
    plt.legend()
    plt.savefig("results/generated_graphs/" + codebase + "_" + lr + ".png")
    plt.show()

def showAverageBetweenSystems(y_index, codebase, lr):
    # all_y = []
    max_length = 0
    for s in systems:
        if lr in results[s][codebase]:
            max_length = max(max_length, len(results[s][codebase][lr]))
        # all_y.append([])

        # for ep in results[s][codebase][lr]:
        #     all_x[-1].append(ep[0])
        #     all_y[-1].append(ep[y_index])
    
    x = list(range(max_length))
    average_y = []
    # y_errors = []

    for i in range(max_length):
        y = []
        for s in systems:
            if lr in results[s][codebase] and len(results[s][codebase][lr]) > i:
                y.append(float(results[s][codebase][lr][i][y_index]))
        average_y.append(mean(y))
        # y_errors.append((max(y) - min(y)) / 2)

    return x, average_y

        

# 0 for episodes, 1 for timestamp
x_index = 0
# 2 for per-episode result, 3 for average of last 100
y_index = 3
# codebase
codebase = "v2"
# learning rates
lrs = ['0.0001', '0.0002', '0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008', '0.0009', '0.001']

# for lr in lrs:
#     showComparisionGraphs(x_index, y_index, codebase, lr)
for lr in lrs:
    x, average_y = showAverageBetweenSystems(y_index, codebase, lr)
    plt.plot(x, average_y, label=lr)

plt.ylim(-650, 250)
plt.xlabel("episode")
plt.ylabel("score")
plt.title("Average of 100 last episodes over every system")
plt.legend()
plt.show()


# plt.plot(list(range(len(scores))), episode_score, average_score)
# plt.savefig(filename + ".png")
import matplotlib.pyplot as plt
import os
import csv 

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
        
# 0 for episodes, 1 for timestamp
x_index = 0

# 2 for per-episode result, 3 for average of last 100
y_index = 3

# codebase
codebase = "v2"

# learning rate
lrs = ['0.0001', '0.0002', '0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008', '0.0009', '0.001']

for lr in lrs:
    for s in results:
        x = []
        y = []
        if lr in results[s][codebase]:
            for i in range(len(results[s][codebase][lr])):
                x_value = results[s][codebase][lr][i][x_index]
                y_value = float(results[s][codebase][lr][i][y_index])

                if x_index == 0:
                    x.append(int(x_value))
                elif x_index == 1:
                    x.append(float(x_value))

                y.append(float(y_value))
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
# plt.plot(list(range(len(scores))), episode_score, average_score)
# plt.savefig(filename + ".png")
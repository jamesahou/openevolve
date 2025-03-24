import matplotlib.pyplot as plt
import numpy as np
with open("/var/local/atharvas/j/funsearch-ks-law/a.err", "r") as f:
    island_lines = []
    time_lines = []

    for line in f:
        if ("absl" in line) and ("backup" not in line):
            island_lines.append(line)
        elif ("root" in line) and ("Iteration" in line):
            time_lines.append(line)

island_scores = {}
for line in island_lines:
    print(line)
    line = line.strip()
    island_id = line.split(" ")[4]
    if island_id not in island_scores:
        island_scores[island_id] = [float(line.split(" ")[-1])]
    else:
        island_scores[island_id].append(float(line.split(" ")[-1]))

times = []
for line in time_lines:
    if float(line.split(" ")[-2]) < 10000:
        times.append(float(line.split(" ")[-2]))

# plt.figure()
# plt.plot(times)
# plt.savefig("time_plot.png")

plt.figure()
for island_id in island_scores:
    plt.plot(-1 * np.array(island_scores[island_id]))
plt.savefig("island_scores-gemini.png")

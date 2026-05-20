import matplotlib.pyplot as plt

gops = [2, 4, 6, 8]
ratios = [50.05, 77.62, 118.68, 170.04]
times = [338, 271, 376, 307]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(gops, ratios, marker='o', color='blue', linewidth=2)
ax1.set_title("Compression Ratio vs GOP")
ax1.set_xlabel("GOP Size")
ax1.set_ylabel("Compression Ratio")
ax1.grid(True)
for g, r in zip(gops, ratios):
    ax1.annotate(str(r), xy=(g, r), xytext=(0, 8), textcoords="offset points", ha='center')

ax2.bar(gops, times, color='salmon', width=1)
ax2.set_title("Execution Time vs GOP")
ax2.set_xlabel("GOP Size")
ax2.set_ylabel("Time (s)")
ax2.grid(True, axis='y')
for g, t in zip(gops, times):
    ax2.annotate(str(t)+'s', xy=(g, t), xytext=(0, 5), textcoords="offset points", ha='center')

plt.suptitle("GOP Analysis (QF=1)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("gop_vs_compression.png", dpi=150)
plt.show()

import matplotlib.pyplot as plt

q_values = [0.5, 1, 2, 4]
ratios = [59.76, 77.62, 88.68, 91.06]
times = [188, 271, 205, 336]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(q_values, ratios, marker='o', color='green', linewidth=2)
ax1.set_title("Compression Ratio vs QF")
ax1.set_xlabel("Q Factor")
ax1.set_ylabel("Compression Ratio")
ax1.grid(True)
for q, r in zip(q_values, ratios):
    ax1.annotate(str(r), xy=(q, r), xytext=(0, 8), textcoords="offset points", ha='center')

ax2.bar(q_values, times, color='lightblue', width=0.8)
ax2.set_title("Execution Time vs QF")
ax2.set_xlabel("Q Factor")
ax2.set_ylabel("Time (s)")
ax2.grid(True, axis='y')
for q, t in zip(q_values, times):
    ax2.annotate(str(t)+'s', xy=(q, t), xytext=(0, 5), textcoords="offset points", ha='center')

plt.suptitle("QF Analysis (GOP=4)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("qf_vs_compression.png", dpi=150)
plt.show()

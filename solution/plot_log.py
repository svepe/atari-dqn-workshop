import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_pickle("training_log.p")

_, axes = plt.subplots(3, 1, sharex=True)
df.plot(kind="line", y="total_reward", ax=axes[0])
df.plot(kind="line", y="average_loss", ax=axes[1])
df.plot(kind="line", y="epsilon", ax=axes[2])
plt.show()

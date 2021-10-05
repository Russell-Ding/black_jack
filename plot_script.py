import matplotlib.pyplot as plt
import pandas as pd


analysis = pd.read_excel(r'Black_Jack_Simulation.xlsx', index_col = 0, engine = 'openpyxl')

fig, axes = plt.subplots(2 )
temp = analysis['random_return'].value_counts()
axes[0].bar(temp.index, temp.values/temp.sum(), width = 0.4)
axes[0].set_title('Random Return')
axes[0].set_xlabel('Return rate')
axes[0].set_ylabel('Frequency')

temp = analysis['strategy_return'].value_counts()
axes[1].bar(temp.index, temp.values/temp.sum(), width = 0.4)
axes[1].set_title('Strategy Return')
axes[1].set_xlabel('Return rate')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2)
temp = analysis['random_action'].fillna(-1).value_counts()
axes[0].bar(temp.index, temp.values/temp.sum(), width = 0.4)
axes[0].set_title('Random Action')
axes[0].set_xlabel('Action')
axes[0].set_ylabel('Frequency')

temp = analysis['strategy_action'].fillna(-1).value_counts()
axes[1].bar(temp.index, temp.values/temp.sum(), width = 0.4)
axes[1].set_title('Strategy Action')
axes[1].set_xlabel('Action')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


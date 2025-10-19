
import pandas as pd
data1 = pd.read_csv(r'C:\Users\91876\Desktop\phish guard\dataset\Assassin.csv')
data2 = pd.read_csv(r'C:\Users\91876\Desktop\phish guard\dataset\spam.csv')
combined = pd.concat([data1, data2], ignore_index=True)

print("Combined dataset preview:")
print(combined.head())
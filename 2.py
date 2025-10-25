import pandas as pd
import numpy as np

data = {
    'hemoglobin': [131, 135, 140, 142, 148, 150, 155, 160, 162, 170],
    'lipoproteins': [4.1, 4.3, 4.0, 4.6, 4.9, 5.0, 5.3, 5.8, 5.7, 6.2]
}

df = pd.DataFrame(data)

print("Наші вихідні дані:")
print(df)
print("-" * 30)


correlation_coefficient = df['hemoglobin'].corr(df['lipoproteins'])

print(f"Коефіцієнт кореляції Пірсона (r) між \n"
      f"гемоглобіном та ліпопротеїнами: {correlation_coefficient:.4f}")


print("\nМатриця кореляцій:")
print(df.corr())
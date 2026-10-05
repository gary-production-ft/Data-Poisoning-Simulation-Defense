import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['target'] = data.target

# Save to local csv
df.to_csv('dataset.csv', index=False)
print("dataset.csv created successfully!")

import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset with 10,000 samples
X, y = make_classification(
    n_samples=10000, 
    n_features=10, 
    n_informative=5, 
    n_redundant=2, 
    n_classes=2, 
    random_state=42
)

# Create a DataFrame
columns = [f'feature_{i+1}' for i in range(10)]
df = pd.DataFrame(X, columns=columns)
df['target'] = y

# Save to local csv
df.to_csv('dataset.csv', index=False)
print("dataset.csv with 10,000 records created successfully!")

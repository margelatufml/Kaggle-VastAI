import pandas as pd

# Load original and extra labeled data
df_kaggle = pd.read_csv('train.csv')
df_aux = pd.read_csv('auxiliary_labeled.csv')   # <-- Change this name to the file you actually have

# Consistency check: column names & author label consistency
print("Kaggle train authors:", sorted(df_kaggle['author'].unique()))
print("Aux authors:", sorted(df_aux['author'].unique()))
assert set(df_kaggle['author'].unique()) == set(df_aux['author'].unique())

# Remove duplicates if any (very rare)
all_df = pd.concat([df_kaggle, df_aux], ignore_index=True)
all_df = all_df.drop_duplicates(subset=['text', 'author'])

print("Combined dataset shape:", all_df.shape)
print(all_df['author'].value_counts())

# Save for next step
all_df.to_csv('train_combined.csv', index=False)

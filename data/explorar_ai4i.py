from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=601)
X = ds.data.features
y = ds.data.targets
print(X.shape)
print(X.columns.tolist())
print(X.head())

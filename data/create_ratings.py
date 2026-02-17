import pandas as pd

# Load IPL stats
df = pd.read_csv("data/players.csv")

# Replace missing values with 0
df.fillna(0, inplace=True)

# Calculate rating
df["rating"] = (
    0.5 * (df["runs"] / 100) +
    0.4 * (df["wickets"]) -
    0.1 * (df["economy"])
)

# Keep only needed columns for auction
auction_df = df[["name", "role", "rating", "base_price"]]

# Save as players.csv for MARL
auction_df.to_csv("data/players.csv", index=False)

print("✅ players.csv created with ratings!")
print(auction_df.head())

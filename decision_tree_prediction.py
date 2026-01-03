import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns

# 1. Load Data (Seaborn has a clean version)
df = sns.load_dataset('titanic')

# 2. Engineering Reality: Trees hate NaNs and Strings (in sklearn implementation)
# We drop simpler columns for this demo to focus on the 'Logic'
df = df[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()

# Convert 'sex' to 0/1 (Label Encoding is fine for binary; OHE is safer generally)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

X = df.drop('survived', axis=1)
y = df['survived']

# 3. Stratified Split (You know why this is important from Mod 2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Build the "Overfit" Monster
# max_depth=None means it will keep splitting until leaves are pure
tree_unrestricted = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
tree_unrestricted.fit(X_train, y_train)

# 5. Evaluate
train_acc = accuracy_score(y_train, tree_unrestricted.predict(X_train))
test_acc = accuracy_score(y_test, tree_unrestricted.predict(X_test))

print(f"----DECISION TREE RESULTS----")
print(f"Training Accuracy: {train_acc:.4f} (Memorization)")
print(f"Test Accuracy:     {test_acc:.4f} (Reality)")
print(f"Tree Depth:        {tree_unrestricted.get_depth()}")
print(f"Number of Leaves:  {tree_unrestricted.get_n_leaves()}")


# Only show the first few levels to keep it readable, 
# but notice how specific the rules get even early on.
#r = export_text(tree_unrestricted, feature_names=list(X.columns), max_depth=5)
#print(r)



#---------TUNED RANDOM FOREST CODE-------------------

rf_tuned = RandomForestClassifier(
    n_estimators=100,       # n_estimators=100: Build 100 trees
    max_depth=5,            # Hard limit on depth (prevents complex nonsense rules)
    min_samples_leaf=5,     # Requires at least 5 people to make a rule
    max_features='sqrt',    # Force diversity (only look at sqrt(#cols) at each split)
    random_state=42,
    n_jobs=-1               # n_jobs=-1: Use all CPU cores (Engineering Reality: Trees are parallelizable!)
)
# 2. Train (This trains 100 trees instantly)
rf_tuned.fit(X_train, y_train)

# 3. Evaluate
train_acc = accuracy_score(y_train, rf_tuned.predict(X_train))
test_acc = accuracy_score(y_test, rf_tuned.predict(X_test))

print("\n--- Tuned Random Forest ---")
print(f"Train Acc: {train_acc:.4f}")
print(f"Test Acc:  {test_acc:.4f}")

# 4. Feature Importance (The "Why")
# Since we can't plot 100 trees, we ask the forest: "Which columns were most useful?"

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_tuned.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# Visualize it
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("What actually mattered?")
plt.show()




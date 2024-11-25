import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("lung_cancer_data.csv")  


data = pd.get_dummies(data, drop_first=True)
X = data.drop("Survival_Months", axis=1)
y = data["Survival_Months"]


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  


results = []


for warm_start in [True, False]:
    for C in [0.1, 1, 10]:
        for penalty in ['l1', 'l2']:
            for max_iter in [100, 500, 1000]:
                print(f"Start of modeling with parameters: Warm start: {warm_start}, C value: {C}, Penalty: {penalty}, Solver: liblinear, Max iter: {max_iter};")
                model = LogisticRegression(warm_start=warm_start, C=C, penalty=penalty, solver='liblinear', max_iter=max_iter, random_state=42)
                
                model.fit(X_train, y_train)
                
                y_pred_val = model.predict(X_val)
                accuracy_val = accuracy_score(y_val, y_pred_val)
                
                results.append([warm_start, C, penalty, 'liblinear', max_iter, accuracy_val])

results_df = pd.DataFrame(results, columns=['Warm_start', 'C', 'Penalty', 'Solver', 'Max_Iter', 'Accuracy'])

results_df.to_csv("model_results.csv", index=False)
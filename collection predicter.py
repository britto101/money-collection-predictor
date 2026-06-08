import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel(
    r"C:\Users\ANTHONY SAMY\Downloads\collection_data.xlsx"
)

print(df.head())

X = df[
    [
        "Amount_Due",
        "Days_Since_Last_Collection",
        "Success_Rate"
    ]
]

y = df["Go_Today"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(
    "Accuracy:",
    round(accuracy_score(y_test, pred) * 100, 2),
    "%"
)

print("\nRecommendations\n")

for _, row in df.iterrows():

    data = pd.DataFrame({
        "Amount_Due": [row["Amount_Due"]],
        "Days_Since_Last_Collection":
            [row["Days_Since_Last_Collection"]],
        "Success_Rate":
            [row["Success_Rate"]]
    })

    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100

    if result == 1:
        print(
            f"{row['Customer']} --> GO TODAY ({prob:.2f}%)"
        )
    else:
        print(
            f"{row['Customer']} --> GO TOMORROW ({100-prob:.2f}%)"
        )

import numpy as np
import pandas as pd
np.random.seed(42)
students = 500
data = {
    "cgpa": np.random.uniform(5, 9.5, students).round(2),
    "aptitude": np.random.randint(30, 100, students),
    "coding": np.random.randint(1, 6, students),
    "communication": np.random.randint(1, 6, students),
    "internships": np.random.randint(0, 4, students),
    "projects": np.random.randint(0, 6, students),
    "backlogs": np.random.randint(0, 6, students)
}
score = (
    data["cgpa"] * 2 +
    data["aptitude"] * 0.05 +
    data["coding"] * 3 +
    data["communication"] * 2 +
    data["internships"] * 4 +
    data["projects"] * 2 -
    data["backlogs"] * 5
)
data["placed"] = (score > 60).astype(int)
df = pd.DataFrame(data)
print(df.head())
X = df.drop("placed", axis=1)
y = df["placed"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
cgpa = float(input("Enter CGPA: "))
aptitude = int(input("Enter Aptitude score (30-100): "))

if aptitude < 30 or aptitude > 100:
    print("⚠️ Aptitude score must be between 30 and 100")
    exit()

coding = int(input("Enter Coding skill (1-5): "))
communication = int(input("Enter Communication skill (1-5): "))
internships = int(input("Enter Internships: "))
projects = int(input("Enter Projects: "))
backlogs = int(input("Enter Backlogs: "))
user = pd.DataFrame([{
    "cgpa": cgpa,
    "aptitude": aptitude,
    "coding": coding,
    "communication": communication,
    "internships": internships,
    "projects": projects,
    "backlogs": backlogs
}])
result = model.predict(user)[0]
prob = model.predict_proba(user)[0][1]

print("\nPrediction:", "Placed" if result == 1 else "Not Placed")
print("Confidence:", round(prob * 100, 2), "%")
print("\nSuggestions:")

if cgpa < 7:
    print("- Improve CGPA")
if aptitude < 60:
    print("- Improve aptitude skills")
if coding < 3:
    print("- Improve coding skills")
if communication < 3:
    print("- Improve communication skills")
if internships == 0:
    print("- Do at least one internship")
if projects < 2:
    print("- Build more projects")
if backlogs > 0:
    print("- Clear backlogs")


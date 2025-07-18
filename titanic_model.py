
# Titanic Survival Prediction - Projeto de Classificação
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# P1 - Carregando e limpando os dados
df = pd.read_csv('train.csv')

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])

# P2 - Separar as variáveis independentes e dependente
X = df.drop('Survived', axis=1)
y = df['Survived']

# P2.1 - Normalização dos dados com StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# P3 - Definir os modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

modelos = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# P4 - Validação cruzada (5-Fold com 30 repetições)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer

f1 = make_scorer(f1_score)
balanced = make_scorer(balanced_accuracy_score)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Avaliar F1 Score
resultados_f1 = {}
for nome, modelo in modelos.items():
    print(f"Treinando (F1): {nome}")
    pontuacoes = []
    for _ in range(30):
        score = cross_val_score(modelo, X_scaled, y, cv=kf, scoring=f1)
        pontuacoes.append(np.mean(score))
    resultados_f1[nome] = pontuacoes

# Avaliar Balanced Accuracy
resultados_balanced = {}
for nome, modelo in modelos.items():
    print(f"Treinando (Balanced): {nome}")
    pontuacoes_bal = []
    for _ in range(30):
        score = cross_val_score(modelo, X_scaled, y, cv=kf, scoring=balanced)
        pontuacoes_bal.append(np.mean(score))
    resultados_balanced[nome] = pontuacoes_bal

# P5 - Visualização dos resultados

# F1 Score - Boxplot
df_f1 = pd.DataFrame(resultados_f1)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_f1)
plt.title("Comparação de F1 Score entre modelos")
plt.ylabel("F1 Score")
plt.grid(True)
plt.show()

# Balanced Accuracy - Boxplot
df_bal = pd.DataFrame(resultados_balanced)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_bal)
plt.title("Comparação de Balanced Accuracy entre modelos")
plt.ylabel("Balanced Accuracy")
plt.grid(True)
plt.show()

# P6 - Resumo estatístico
print("\nResumo das médias (F1 Score):")
for nome in df_f1.columns:
    media = np.mean(df_f1[nome])
    desvio = np.std(df_f1[nome])
    print(f"{nome}: {media:.4f} ± {desvio:.4f}")

print("\nResumo das médias (Balanced Accuracy):")
for nome in df_bal.columns:
    media = np.mean(df_bal[nome])
    desvio = np.std(df_bal[nome])
    print(f"{nome}: {media:.4f} ± {desvio:.4f}")


# Titanic Survival Prediction - Projeto de Classificação com Análise do SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# P1 - Carregando e limpando os dados
df = pd.read_csv('train.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# P3 - Modelos
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

# Validação cruzada
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer

f1 = make_scorer(f1_score)
balanced = make_scorer(balanced_accuracy_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados_f1 = {}
resultados_balanced = {}

for nome, modelo in modelos.items():
    print(f"Treinando: {nome}")
    f1_scores = []
    bal_scores = []
    for _ in range(30):
        f1_cv = cross_val_score(modelo, X_scaled, y, cv=kf, scoring=f1)
        bal_cv = cross_val_score(modelo, X_scaled, y, cv=kf, scoring=balanced)
        f1_scores.append(np.mean(f1_cv))
        bal_scores.append(np.mean(bal_cv))
    resultados_f1[nome] = f1_scores
    resultados_balanced[nome] = bal_scores

# P5 - Boxplots gerais
df_f1 = pd.DataFrame(resultados_f1)
df_bal = pd.DataFrame(resultados_balanced)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_f1)
plt.title("Comparação de F1 Score entre modelos")
plt.ylabel("F1 Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_bal)
plt.title("Comparação de Balanced Accuracy entre modelos")
plt.ylabel("Balanced Accuracy")
plt.grid(True)
plt.show()

# P6 - Estatísticas
print("\nResumo das médias (F1 Score):")
for nome in df_f1.columns:
    print(f"{nome}: {np.mean(df_f1[nome]):.4f} ± {np.std(df_f1[nome]):.4f}")

print("\nResumo das médias (Balanced Accuracy):")
for nome in df_bal.columns:
    print(f"{nome}: {np.mean(df_bal[nome]):.4f} ± {np.std(df_bal[nome]):.4f}")

# P7 - Gráficos específicos para o SVM
# Histograma do F1 Score do SVM
plt.figure(figsize=(8, 5))
plt.hist(resultados_f1['SVM'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribuição dos F1 Scores - SVM')
plt.xlabel('F1 Score')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

# Linha de tendência F1 do SVM
plt.figure(figsize=(8, 5))
plt.plot(resultados_f1['SVM'], marker='o')
plt.title('F1 Score nas 30 execuções - SVM')
plt.xlabel('Execução')
plt.ylabel('F1 Score')
plt.grid(True)
plt.show()

# Comparação SVM com Random Forest e Gradient Boosting
df_comparativo = df_f1[['SVM', 'RandomForest', 'GradientBoosting']]
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_comparativo)
plt.title("Comparação detalhada - SVM vs RF vs GB")
plt.ylabel("F1 Score")
plt.grid(True)
plt.show()

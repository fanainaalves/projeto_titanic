# 🛳️ Titanic Survival Prediction - Projeto de Machine Learning

Este projeto tem como objetivo prever a sobrevivência de passageiros do Titanic utilizando técnicas de aprendizado de máquina supervisionado, baseado no conjunto de dados disponibilizado pelo Kaggle.

## 🎯 Objetivo
Desenvolver e comparar modelos de classificação capazes de prever se um passageiro sobreviveu ao naufrágio, com base em atributos como idade, sexo, classe social, tarifa, etc.

## 📁 Dataset
- Fonte: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
- Arquivo principal utilizado: `train.csv`

## ⚙️ Tecnologias e Bibliotecas
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

## 🤖 Modelos Avaliados
Foram utilizados e comparados os seguintes modelos de classificação:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Gradient Boosting

## 🧪 Metodologia
- Aplicação de **StandardScaler** para normalizar os dados de entrada.
- Utilização de **validação cruzada com 5 folds**, repetida **30 vezes**.
- Cálculo da **média** das métricas em cada repetição.
- Comparação entre modelos por **boxplot** e **média ± desvio padrão**.

## 📏 Métricas de Avaliação
- **F1 Score**
- **Acurácia Balanceada (Balanced Accuracy)**

Essas métricas são recomendadas para problemas de classificação e foram exigidas como critério de avaliação da disciplina.

## 📊 Resultados
Os resultados dos modelos são exibidos graficamente por meio de boxplots e resumidos estatisticamente. Os modelos com melhor desempenho foram Random Forest e Gradient Boosting.

## 🚀 Execução
### 1. Instale as dependências:
```bash
pip install -r requirements.txt
```

### 2. Certifique-se de ter o arquivo `train.csv` na mesma pasta do script.

### 3. Execute o código:
```bash
python titanic_model.py
```

## 📂 Estrutura do Projeto
```
├── titanic_model.py   # Código principal com validação cruzada
├── requirements.txt              # Bibliotecas necessárias
├── README.md                     # Instruções do projeto
├── train.csv                     # Base de dados (adquirida no Kaggle)
```

## 👥 Equipe
- Nome da dupla: Fanaina Alves & Eduarda Queiroz.

## 📌 Notas Finais
Este projeto foi desenvolvido como parte da disciplina de Reconhecimento de Padrões. O código será submetido via GitHub e o artigo técnico será entregue em formato PDF seguindo o template oficial da turma.
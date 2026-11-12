# Classificação Estelar com Machine Learning (SDSS17)

Modelo de Machine Learning para classificar objetos astronômicos em **Estrelas (STAR)**, **Galáxias (GALAXY)** e **Quasares (QSO)**, usando dados fotométricos e o redshift do levantamento Sloan Digital Sky Survey (SDSS, DR17).

## Objetivo

Automatizar a rotulagem de grandes volumes de dados astronômicos. Em vez de análise manual objeto a objeto, o modelo identifica a natureza física de cada corpo celeste a partir de suas medições de brilho e desvio para o vermelho.

## Dataset

- **Fonte:** [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17?resource=download) (Kaggle), originado do SDSS DR17.
- **Registros:** 100.000 observações, 18 colunas originais.
- **Variável-alvo:** `class` (STAR, GALAXY, QSO), problema de classificação multiclasse.
- **Features principais:** filtros fotométricos `u`, `g`, `r`, `i`, `z` e o `redshift`.
- **Distribuição das classes:** GALAXY 59,4%, STAR 21,6%, QSO 19,0%.

> O arquivo `star_classification.csv` não está versionado neste repositório. Baixe-o no Kaggle e coloque-o na pasta de dados antes de executar o notebook.

## Metodologia

1. **Análise exploratória (EDA):** verificação de tipos, nulos (nenhum) e duplicados (nenhum); análise do desbalanceamento de classes; identificação de um valor de erro instrumental (`u = -9999`) e análise do redshift por classe.
2. **Preparação:**
   - Remoção do registro com `u = -9999` (código de erro do instrumento).
   - Remoção de colunas de identificação e metadados (`obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`), para evitar vazamento de informação e overfitting.
   - Codificação do alvo com `LabelEncoder` (GALAXY=0, QSO=1, STAR=2).
3. **Divisão treino/teste:** 80/20 com `stratify=y` e `random_state=42` (79.999 treino e 20.000 teste).
4. **Modelos** (ambos em `Pipeline` do scikit-learn, com `StandardScaler`):
   - **Regressão Logística:** baseline linear.
   - **Random Forest** (100 árvores): modelo final.
5. **Avaliação:** matriz de confusão, `classification_report` (precisão, recall, F1) e acurácia. A acurácia sozinha não basta por causa do desbalanceamento, então o F1 e o recall da classe QSO têm peso maior.
6. **Persistência:** o pipeline completo da Random Forest é salvo com `joblib` (`pipeline_classificacao_astronomica.joblib`), e a fidelidade do modelo recarregado foi validada (100% de predições idênticas no conjunto de teste).

## Resultados

| Modelo | Acurácia (teste) |
|---|---|
| Regressão Logística (baseline) | 95,8% |
| **Random Forest (modelo final)** | **97,9%** |

- Na classe QSO, o F1-Score ficou em torno de 0,95 para a Random Forest e 0,91 para a Regressão Logística.
- O macro F1-Score ficou em 0,976 (Random Forest) e 0,952 (Regressão Logística).
- Não há indícios de overfitting: o desempenho de treino e teste ficou próximo.
- A classe mais difícil é QSO, cuja assinatura espectral pode se confundir com a de galáxias elípticas próximas.

## Limitações

- O modelo depende fortemente do **redshift**. Se essa medição for imprecisa, o desempenho tende a cair.
- A classe QSO é a minoritária e a mais difícil de separar.
- Os resultados vêm de um único conjunto de teste; não há validação cruzada documentada.

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/devRenanPaulino/ml-stellar-classification.git
   cd ml-stellar-classification
   ```
2. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```
3. Coloque `star_classification.csv` na pasta de dados (ou ajuste o caminho no notebook).
4. Abra o notebook no Google Colab ou no Jupyter e execute as células em ordem.

Para carregar o modelo salvo:

```python
import joblib

modelo = joblib.load("pipeline_classificacao_astronomica.joblib")
predicoes = modelo.predict(X_novos)  # X_novos: mesmas colunas de features usadas no treino
```

## Estrutura sugerida

```
ml-stellar-classification/
├── README.md
├── notebook/              # Notebook do projeto (Colab/Jupyter)
├── data/                  # star_classification.csv (não versionado)
└── models/                # pipeline_classificacao_astronomica.joblib
```

## Tecnologias

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

## Autor

[Renan Paulino](https://github.com/devRenanPaulino)

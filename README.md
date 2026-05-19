# Tabas — Base Price Modeling for Short-Term Rentals

## Contexto e problema

Plataformas de aluguel de temporada como o Airbnb exibem preços que variam continuamente com sazonalidade, ocupação, eventos locais e estratégias dinâmicas dos anfitriões. Para a **Tabas** — empresa que opera imóveis de alto padrão no Brasil — é essencial distinguir o **preço estrutural** de um imóvel (aquele derivado de seus atributos físicos e localização) da flutuação dinâmica de curto prazo.

O objetivo deste case é construir um modelo capaz de estimar esse **preço-base**, a partir de dados históricos de cotações do Airbnb Brasil, respondendo às perguntas:

- Quanto vale estruturalmente um imóvel, independente da sazonalidade?
- Quais atributos (quartos, localização, avaliações) mais explicam o preço?
- Como entregar essa estimativa de forma reproduzível e pronta para produção?

**Abordagem adotada:** as cotações são filtradas por uma janela de estadia de 15 noites (`LOS=15`) — escolhida por maximizar a cobertura e estabilidade dos dados — e agregadas por imóvel via **mediana**. O alvo `log1p(base_price)` é então modelado com uma **Regressão Ridge** implementada analiticamente em NumPy puro, com features físicas, de localização, de qualidade percebida e one-hot de bairro.

---

## Instalação e configuração

### Pré-requisitos

- Python 3.11+
- Os arquivos `airbnb_apart.csv` e `airbnb_prices.csv` devem estar na pasta `data/`

### Com Conda

```bash
conda create -n tabas-airbnb python=3.11 -y
conda activate tabas-airbnb
pip install -r requirements.txt
python -m ipykernel install --user --name tabas-airbnb --display-name "Python (tabas-airbnb)"
```

### Com venv (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name tabas-airbnb --display-name "Python (tabas-airbnb)"
```

### Com venv (macOS / Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name tabas-airbnb --display-name "Python (tabas-airbnb)"
```

### Dependências principais

| Pacote | Versão mínima | Uso |
|--------|--------------|-----|
| `pandas` | ≥ 2.0 | Manipulação de dados |
| `numpy` | ≥ 1.24 | Álgebra linear (Ridge fechado) |
| `matplotlib` | ≥ 3.7 | Visualizações estáticas |
| `plotly` | ≥ 5.18 | Gráficos interativos |
| `shap` | ≥ 0.44 | Interpretabilidade SHAP |
| `jupyterlab` | ≥ 4.0 | Ambiente de execução |

### Executando o notebook

```bash
jupyter lab tabas_airbnb_base_price_modeling.ipynb
```

Execute **Kernel → Restart & Run All** para reprodução completa.

> Se os CSVs estiverem em outro local, altere a variável `DATA_DIR` na primeira célula de configuração do notebook.

---

## Estrutura do projeto

```
Tabas - Case/
├── tabas_airbnb_base_price_modeling.ipynb   # Notebook principal
├── requirements.txt                         # Dependências Python
├── data/
│   ├── airbnb_apart.csv                     # Cadastro dos imóveis (~3,9 MB)
│   └── airbnb_prices.csv                    # Cotações de preço (~180 MB)
├── models/
│   └── tabas_base_price_model_artifact.json # Artefato do modelo treinado
└── docs/
    ├── tabas_case_presentation.html         # Apresentação executiva interativa
    └── Tabas - Data Scientist Case Study (1).docx  # Enunciado original do case
```

---

## Notebook principal

### `tabas_airbnb_base_price_modeling.ipynb`

| # | Seção | O que faz |
|---|-------|-----------|
| 1 | Enquadramento do problema | Define o objetivo: isolar o preço estrutural, excluindo sazonalidade e dinâmica de ocupação |
| 2 | Configurações e constantes | `PRIMARY_LOS=15`, `ENCODING='latin1'`, `RANDOM_SEED=42` |
| 3 | Funções auxiliares | Leitura robusta dos CSVs, coerção de tipos, métricas customizadas (MAE, RMSE, MAPE, MdAPE, R²) |
| 4 | Carregamento e auditoria | Lê os dois CSVs, cruza IDs, detecta anomalias estruturais |
| 5 | EDA — distribuição de preços | Audita assimetria dos preços, valida escolha do `LOS=15` como janela primária |
| 6 | EDA — bairro e atributos | Sanidade check: bairros de praia têm mediana 8–9× maior que bairros periféricos |
| 7 | Construção do alvo estrutural | Filtro `LOS=15` → remoção de extremos (p0.5%, p99.5%) → mediana por ID → `log1p(base_price)` |
| 8 | Feature Engineering | Ratios de densidade, `log_review_count`, one-hot de bairro (N ≥ 25) |
| 9 | Modelo Ridge | Validação holdout por `id` (80/20), busca de `alpha` em grade, pesos por `√n_cotações` |
| 10 | Avaliação | Métricas globais + segmentação por quartil de preço e por bairro |
| 11 | Interpretação dos drivers | Coeficientes log-lineares → impacto percentual por atributo físico |
| 12 | Função de inferência para produção | `predict_base_price(df)` — pipeline completo em uma chamada |
| 13 | Análise SHAP | Beeswarm plot, importância global (mean \|SHAP\|) e dependência de `bedrooms` |
| 14 | Limitações e próximos passos | LOS=15, ausência de amenidades, regressão à média no luxo, roadmap |

### Resultados (conjunto de teste — 1.759 imóveis)

| Métrica | Valor |
|---------|-------|
| MAE | R$ 240,92 |
| RMSE | R$ 711,32 |
| MdAPE | 28,20% |
| MAPE | 35,47% |
| R² | 0,122 |

---

## Artefato do modelo

### `models/tabas_base_price_model_artifact.json`

JSON serializável com o vetor de pesos `beta`, parâmetros do pré-processador e nomes das features. Permite inferência em produção **sem reexecutar o notebook**.

| Parâmetro | Valor |
|-----------|-------|
| `best_alpha` | 300,0 |
| `primary_los` | 15 dias |
| Features numéricas | 17 |
| Features categóricas | 6 grupos (42 bairros + outros) |
| Total de parâmetros | 72 |

**Carregamento:**

```python
import json, numpy as np

with open("models/tabas_base_price_model_artifact.json") as f:
    artifact = json.load(f)

beta          = np.array(artifact["beta"])
feature_names = artifact["feature_names"]
preprocessor  = artifact["preprocessor"]  # médias, stds, medianas, categorias OHE
primary_los   = artifact["primary_los"]   # 15
```

---

## Apresentação executiva

### `docs/tabas_case_presentation.html`

Documento HTML standalone que narra o projeto com animações e gráficos interativos. Abra diretamente no navegador:

```bash
start docs/tabas_case_presentation.html   # Windows
open  docs/tabas_case_presentation.html   # macOS / Linux
```

---

## Notas técnicas

- Os CSVs usam encoding `latin1` (acentos de bairros brasileiros) — tratado em `read_apartments` e `read_prices`.
- Linhas malformadas no CSV de imóveis são ignoradas via `on_bad_lines="skip"` (reprodutível e auditável).
- O modelo Ridge usa solução analítica fechada em `numpy` puro — `scikit-learn` **não é obrigatório**.
- A seção SHAP requer `shap >= 0.44`; se não instalado, o bloco exibe aviso e o restante continua normalmente.
- Datas (`check_in`, `check_out`, `reference_date`, `extraction_date`) são usadas **apenas para limpeza e agregação** — não entram como features do modelo.# Tabas_case

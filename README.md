# Analise de Risco de Credito

Projeto de estudo para analise exploratoria, modelagem e organizacao de pipeline de dados para risco de credito a partir do dataset **Credit Risk Dataset**, publicado no Kaggle por **Lao Tse**.

Fonte oficial dos dados:
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Segundo a pagina do dataset no Kaggle, a base "contains columns simulating credit bureau data" e esta disponivel sob licenca **CC0: Public Domain**.

## Objetivo

Este repositorio foi evoluido para cobrir duas frentes:

- estudo analitico com notebooks;
- base reutilizavel de engenharia para ingestao, armazenamento e leitura de dados via MinIO.

O MinIO passou a ser a fonte unificada de leitura dos notebooks e do codigo Python. O arquivo local `data/credit_risk_dataset.csv` funciona apenas como staging opcional da ingestao.

## Estrutura do projeto

```text
credit_risk_v2/
|-- src/
|   `-- credit_risk/
|       |-- api/
|       |-- data/
|       |-- datasets/
|       |-- models/
|       |-- serving/
|       |-- storage/
|       |-- training/
|       |-- cli.py
|       |-- config.py
|       `-- logging_utils.py
|-- scripts/
|   |-- ingest_credit_risk_dataset.py
|   `-- read_credit_risk_dataset.py
|-- tests/
|   `-- unit/
|-- notebook/
|   |-- estatistica.ipynb
|   |-- probabilidades_inferencias.ipynb
|   |-- modelagem_risco_credito.ipynb
|   |-- modelagem_risco_credito_feature_selection_automl.ipynb
|   `-- modelagem_risco_credito_automl_optuna_top3.ipynb
|-- data/
|   `-- credit_risk_dataset.csv
|-- docker-compose.yml
|-- .env.example
|-- pyproject.toml
|-- requirements.txt
|-- sitecustomize.py
`-- README.md
```

## Dataset

Arquivo principal:

- `data/credit_risk_dataset.csv`

Resumo da base:

- 12 colunas;
- 32.581 registros;
- target binario em `loan_status`.

### Dicionario de dados

| Coluna | Descricao |
| --- | --- |
| `person_age` | Idade da pessoa |
| `person_income` | Renda anual |
| `person_home_ownership` | Situacao de moradia |
| `person_emp_length` | Tempo de emprego em anos |
| `loan_intent` | Finalidade do emprestimo |
| `loan_grade` | Classificacao do emprestimo |
| `loan_amnt` | Valor do emprestimo |
| `loan_int_rate` | Taxa de juros |
| `loan_status` | Status do emprestimo, onde `0` indica nao inadimplente e `1` indica inadimplente |
| `loan_percent_income` | Percentual da renda comprometido com o emprestimo |
| `cb_person_default_on_file` | Historico de inadimplencia em arquivo |
| `cb_person_cred_hist_length` | Tempo de historico de credito |

## Instalacao

### 1. Criar e ativar ambiente virtual

No Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar o projeto

Fluxo recomendado:

```powershell
python -m pip install --upgrade pip
pip install -e ".[notebooks]"
```

Alternativa:

```powershell
pip install -r requirements.txt
```

O projeto usa layout `src/`, `pyproject.toml` e entry points instalaveis. Isso simplifica imports, execucao de scripts e testes.

## MinIO

### 1. Configurar variaveis

Copie `.env.example` para `.env` e ajuste se necessario.

Variaveis principais:

- `MINIO_ENDPOINT`: endpoint usado pela aplicacao Python
- `MINIO_ACCESS_KEY`: usuario raiz do MinIO
- `MINIO_SECRET_KEY`: senha raiz do MinIO
- `MINIO_BUCKET`: bucket padrao do projeto
- `MINIO_SECURE`: `false` para ambiente local com HTTP
- `MINIO_API_PORT`: porta da API S3
- `MINIO_CONSOLE_PORT`: porta do console web

### 2. Subir os servicos

```powershell
docker compose up -d
```

Logs:

```powershell
docker compose logs -f minio
```

Parar:

```powershell
docker compose down
```

Endpoints padrao:

- API S3: `http://localhost:9000`
- Console: `http://localhost:9001`

Bucket padrao:

- `credit-risk`

## Ingestao do dataset

### CLI

Publicar o dataset no MinIO:

```powershell
credit-risk-ingest
```

Com mes especifico:

```powershell
credit-risk-ingest --month 2026-04
```

Forcando overwrite:

```powershell
credit-risk-ingest --month 2026-04 --force
```

Alternativa via script:

```powershell
python .\scripts\ingest_credit_risk_dataset.py
```

### Comportamento da ingestao

Fluxo da ingestao:

- se `data/credit_risk_dataset.csv` existir, ele e usado como staging local;
- se o arquivo local nao existir, o projeto baixa a base do Kaggle via `kagglehub`;
- se o objeto mensal ja existir no MinIO, o CSV nao e reenviado;
- se o objeto existir mas o manifesto estiver ausente ou stale, apenas o `latest.json` e atualizado.

Estrutura de armazenamento do dataset:

```text
datasets/raw/credit_risk_dataset/YYYY-MM/credit_risk_dataset.csv
```

Manifesto da ultima versao:

```text
_catalog/datasets/raw/credit_risk_dataset/latest.json
```

## Leitura unificada do dataset

### CLI

```powershell
credit-risk-read
```

Com mes especifico:

```powershell
credit-risk-read --month 2026-04
```

### Uso em Python

```python
from credit_risk.datasets import load_credit_risk_dataset

df = load_credit_risk_dataset()
df_mes_fechado = load_credit_risk_dataset(month="2026-04")
```

Essa leitura usa o MinIO como fonte oficial e resolve o dataset pelo manifesto `latest.json` quando nenhum mes e informado.

## API FastAPI

### Subir a API

```powershell
credit-risk-api
```

Alternativa com `python -m uvicorn`:

```powershell
python -m uvicorn credit_risk.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Se estiver usando o ambiente virtual local, prefira:

```powershell
.\venv\Scripts\python.exe -m uvicorn credit_risk.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Observacao para Windows corporativo:

- em alguns ambientes, `uvicorn.exe` ou outros launchers `.exe` criados pelo `pip` podem ser bloqueados por politica de seguranca;
- nesses casos, use `python -m uvicorn ...`, que normalmente evita esse bloqueio.

### Endpoints

- API base: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Healthcheck: `GET /health`
- Ingestao: `POST /datasets/credit-risk/ingest`
- Treino: `POST /models/credit-risk/train`
- Predicao: `POST /models/credit-risk/predict`

### Exemplo de payload

```json
{
  "local_path": "data/credit_risk_dataset.csv",
  "month": "2026-04",
  "overwrite": false
}
```

### Exemplo com curl

```bash
curl -X POST "http://localhost:8000/datasets/credit-risk/ingest" \
  -H "Content-Type: application/json" \
  -d "{\"local_path\":\"data/credit_risk_dataset.csv\",\"month\":\"2026-04\",\"overwrite\":false}"
```

### Treinar o modelo de quorum Top 3

O projeto implementa um pipeline de treino baseado nos notebooks:

- split estratificado em treino, teste e implementacao;
- feature engineering de dominio;
- imputacao e normalizacao para atributos numericos;
- imputacao e one-hot encoding para categoricos;
- selecao de features com `mutual_info` e `RandomForest`;
- benchmark entre familias de modelos;
- tuning com Optuna nos 3 melhores;
- ensemble por quorum usando a media das probabilidades dos 3 melhores modelos.

O treinamento so persiste o modelo se a acuracia no conjunto de implementacao for maior ou igual ao minimo configurado. O padrao da API e `0.80`.

Payload de treino:

```json
{
  "dataset_month": "2026-04",
  "model_name": "credit_risk_quorum_top3",
  "minimum_accuracy": 0.80,
  "optuna_trials": 10,
  "threshold": 0.50,
  "model_month": "2026-04"
}
```

Exemplo com `curl`:

```bash
curl -X POST "http://localhost:8000/models/credit-risk/train" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_month\":\"2026-04\",\"model_name\":\"credit_risk_quorum_top3\",\"minimum_accuracy\":0.80,\"optuna_trials\":10,\"threshold\":0.50,\"model_month\":\"2026-04\"}"
```

### Predicao em nivel de negocio

A rota de predicao recebe os dados brutos da solicitacao de credito, carrega o modelo versionado do MinIO e devolve:

- probabilidade de default;
- classe prevista;
- faixa de risco;
- recomendacao de negocio;
- razoes heuristicas para a decisao.

Payload de exemplo:

```json
{
  "person_age": 32,
  "person_income": 25000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 12000,
  "loan_int_rate": 17.5,
  "loan_percent_income": 0.42,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 6,
  "model_name": "credit_risk_quorum_top3",
  "model_month": "2026-04"
}
```

Exemplo com `curl`:

```bash
curl -X POST "http://localhost:8000/models/credit-risk/predict" \
  -H "Content-Type: application/json" \
  -d "{\"person_age\":32,\"person_income\":25000,\"person_home_ownership\":\"RENT\",\"person_emp_length\":3,\"loan_intent\":\"PERSONAL\",\"loan_grade\":\"C\",\"loan_amnt\":12000,\"loan_int_rate\":17.5,\"loan_percent_income\":0.42,\"cb_person_default_on_file\":\"Y\",\"cb_person_cred_hist_length\":6,\"model_name\":\"credit_risk_quorum_top3\",\"model_month\":\"2026-04\"}"
```

## Notebooks

Os notebooks foram ajustados para:

- adicionar `src/` ao `sys.path` de forma robusta;
- importar `load_credit_risk_dataset` do pacote;
- ler o dataset do MinIO em vez de acessar CSV local diretamente.

Fluxo recomendado para notebooks:

1. ativar o ambiente virtual;
2. subir o MinIO;
3. publicar o dataset com `credit-risk-ingest`;
4. iniciar o Jupyter;
5. abrir um notebook em `notebook/`;
6. executar as celulas em ordem.

Iniciar Jupyter:

```powershell
jupyter notebook
```

ou

```powershell
jupyter lab
```

## Persistencia de modelos

O projeto tambem suporta armazenamento de modelos no MinIO.

Estrutura:

```text
models/trained/<model_name>/<YYYY-MM>/<artifact_name>
_catalog/models/trained/<model_name>/latest.json
```

Artefatos gerados pelo treino:

- `model.pkl`: bundle completo do modelo de quorum com preprocessing, selecao de features e ensemble;
- `training_report.json`: resumo versionado com ranking, metricas e hiperparametros.

Exemplo:

```python
from sklearn.ensemble import RandomForestClassifier

from credit_risk.models import ModelRegistry

model = RandomForestClassifier(random_state=42)
registry = ModelRegistry()

registry.save_pickle(
    model=model,
    model_name="random_forest_credit_risk",
    artifact_name="model.pkl",
)
```

## Testes

Executar suite unitria:

```powershell
python -m unittest discover -s tests\unit -v
```

Cobertura principal:

- idempotencia do upload mensal do dataset;
- fallback local/Kaggle;
- manifestos `latest.json`;
- persistencia e leitura de modelos;
- rota HTTP de ingestao quando `fastapi` estiver disponivel no ambiente.

## Arquitetura resumida

Componentes principais em `src/credit_risk`:

- `api/`: FastAPI, Swagger e rota de ingestao
- `serving/`: carregamento do modelo e resposta em linguagem de negocio
- `cli.py`: entry points de linha de comando
- `config.py`: configuracao via ambiente
- `storage/`: cliente MinIO e convencoes de object paths
- `data/catalog.py`: catalogo versionado de datasets
- `datasets/credit_risk.py`: API de alto nivel para o dataset principal
- `models/registry.py`: persistencia de modelos e artefatos
- `training/`: treino, feature engineering, benchmark, tuning e modelo de quorum
- `logging_utils.py`: configuracao simples de logging

## Observacoes

- `data/` pode continuar sendo usado como staging local, mas nao deve ser tratado como fonte oficial do pipeline.
- Se o MinIO nao estiver no ar ou se o dataset ainda nao tiver sido publicado, os notebooks nao vao carregar a base.
- `pyproject.toml` e a fonte principal de dependencias do projeto.
- `requirements.txt` foi mantido como atalho para instalar `-e .[notebooks]`.

## Proximos passos sugeridos

- expor rotas para leitura de manifestos e listagem de datasets/modelos;
- adicionar persistencia de datasets tratados e features prontas para treino;
- publicar artefatos de avaliacao e metricas no MinIO;
- integrar treino e inferencia em servicos HTTP separados.

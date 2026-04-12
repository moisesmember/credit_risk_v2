# Análise de Risco de Crédito

Projeto de estudo para analise exploratoria e preparacao de dados de risco de credito a partir do dataset **Credit Risk Dataset**, publicado no Kaggle por **Lao Tse**.

Fonte oficial dos dados:
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Segundo a pagina do dataset no Kaggle, a base "contains columns simulating credit bureau data" e esta disponivel sob licenca **CC0: Public Domain**.

## Objetivo

Este repositorio organiza um ambiente simples para explorar uma base tabular de risco de credito com foco em:

- compreensao das variaveis da base;
- analise estatistica inicial;
- apoio a futuras etapas de modelagem preditiva;
- reproducao do fluxo em **Jupyter Notebook**.

Os notebooks do projeto agora usam o MinIO como fonte unificada de dados. O arquivo local `data/credit_risk_dataset.csv` funciona apenas como staging opcional para a etapa de ingestao.

## Estrutura do projeto

```text
credit_risk_v2/
|-- src/
|   `-- credit_risk/
|       |-- cli.py
|       |-- config.py
|       |-- data/
|       |-- datasets/
|       |-- models/
|       `-- storage/
|-- scripts/
|   |-- ingest_credit_risk_dataset.py
|   `-- read_credit_risk_dataset.py
|-- data/
|   `-- credit_risk_dataset.csv
|-- notebook/
|   `-- estatistica.ipynb
|-- docker-compose.yml
|-- .env.example
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Dataset

O arquivo local utilizado no projeto e `data/credit_risk_dataset.csv`.

Com base no arquivo presente neste repositorio, a base possui:

- 12 colunas;
- 32.581 registros de dados;
- variavel alvo binaria em `loan_status`.

### Dicionario de dados

As descricoes abaixo foram organizadas a partir da pagina do dataset no Kaggle:

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

## Como executar

### 1. Criar e ativar um ambiente virtual

No Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar o projeto

Opcao recomendada com `pyproject.toml`:

```powershell
python -m pip install --upgrade pip
pip install -e ".[notebooks]"
```

Se preferir manter o fluxo anterior:

```powershell
pip install -r requirements.txt
```

O projeto usa layout `src/` e passa a funcionar como pacote instalavel. Isso facilita:

- imports consistentes em notebooks, scripts e testes;
- entry points de linha de comando;
- reutilizacao do pipeline fora da raiz do repositorio.

### 3. Subir o MinIO

Copie `.env.example` para `.env` e ajuste se necessario.

Depois, inicie o MinIO:

```powershell
docker compose up -d
```

Para acompanhar os logs do container:

```powershell
docker compose logs -f minio
```

Para parar os servicos:

```powershell
docker compose down
```

Endpoints padrao:

- API S3: `http://localhost:9000`
- Console: `http://localhost:9001`

Bucket padrao:

- `credit-risk`

Variaveis principais do `.env`:

- `MINIO_ENDPOINT`: endpoint usado pela aplicacao Python;
- `MINIO_ACCESS_KEY`: usuario raiz do MinIO;
- `MINIO_SECRET_KEY`: senha raiz do MinIO;
- `MINIO_BUCKET`: bucket padrao do projeto;
- `MINIO_SECURE`: `false` para ambiente local com HTTP.

### 4. Publicar o dataset do Kaggle no MinIO

```powershell
credit-risk-ingest
```

Fluxo do script:

- se `data/credit_risk_dataset.csv` ja existir, ele usa esse arquivo como staging local;
- se o arquivo local nao existir, ele baixa o dataset do Kaggle via `kagglehub`;
- se o objeto mensal ja existir no MinIO, ele nao faz novo upload do CSV;
- se necessario, ele apenas atualiza o manifesto `latest.json`.

Esse passo precisa ser executado antes de abrir os notebooks, porque eles leem o dataset a partir do MinIO.

Alternativa sem instalar o entry point:

```powershell
python .\scripts\ingest_credit_risk_dataset.py
```

Por padrao, o dataset sera salvo no bucket configurado com a estrutura:

```text
datasets/raw/credit_risk_dataset/YYYY-MM/credit_risk_dataset.csv
```

O manifesto do ultimo upload fica em:

```text
_catalog/datasets/raw/credit_risk_dataset/latest.json
```

Para sobrescrever o objeto do mes explicitamente:

```powershell
credit-risk-ingest --force
```

### 5. Ler o dataset de forma unificada a partir do MinIO

Exemplo rapido:

```powershell
credit-risk-read
```

Uso em Python:

```python
from credit_risk.datasets import load_credit_risk_dataset

df = load_credit_risk_dataset()
df_mes_fechado = load_credit_risk_dataset(month="2026-04")
```

Alternativa via script:

```powershell
python .\scripts\read_credit_risk_dataset.py --month 2026-04
```

### 6. Iniciar o Jupyter

```powershell
jupyter notebook
```

Se preferir, voce tambem pode usar:

```powershell
jupyter lab
```

### 7. Abrir o notebook do projeto

No navegador, abra:

```text
http://localhost:8888/tree
```

Em algumas execucoes, o Jupyter exibira no terminal um link com token, por exemplo:

```text
http://localhost:8888/tree?token=...
```

Depois, navegue ate:

```text
notebook/estatistica.ipynb
```

Todos os notebooks foram ajustados para:

- adicionar `src/` ao `sys.path` de forma robusta;
- importar `load_credit_risk_dataset` do pacote do projeto;
- ler o dataset do MinIO, em vez de acessar `../data/credit_risk_dataset.csv` diretamente.

### 8. Selecionar o kernel Python

Dentro do Jupyter, selecione o kernel do ambiente virtual criado para garantir que as dependencias instaladas em `requirements.txt` sejam utilizadas.

## Uso com Jupyter

Este projeto foi organizado para ser executado principalmente no **Jupyter Notebook**.

Fluxo esperado:

1. ativar o ambiente virtual;
2. subir o MinIO;
3. publicar o dataset com `credit-risk-ingest`;
4. iniciar o Jupyter;
5. abrir `notebook/estatistica.ipynb`;
6. executar as celulas em ordem.

O projeto agora tambem possui uma camada reutilizavel para centralizar datasets e artefatos no MinIO, evitando que cada notebook leia arquivos locais de forma diferente.

## Pipeline reutilizavel com MinIO

A estrutura em `src/credit_risk` foi organizada para servir como base de pipeline:

- `cli.py`: entry points de linha de comando para ingestao e leitura;
- `config.py`: configuracao de conexao ao MinIO via variaveis de ambiente;
- `storage/`: cliente MinIO e convencoes de paths para datasets, catalogos e modelos;
- `data/catalog.py`: publicacao e leitura versionada de datasets;
- `logging_utils.py`: configuracao simples de logs para scripts e pipeline;
- `datasets/credit_risk.py`: API unificada para o dataset principal do projeto;
- `models/registry.py`: persistencia de modelos e outros artefatos de ML.

### Como os notebooks leem dados

Exemplo do padrao agora usado nos notebooks:

```python
from credit_risk.datasets import load_credit_risk_dataset

df = load_credit_risk_dataset()
```

Isso faz a leitura a partir do manifesto `latest.json` no MinIO. Se quiser analisar um mes especifico:

```python
df = load_credit_risk_dataset(month="2026-04")
```

### Convencao de armazenamento

Datasets brutos:

```text
datasets/raw/<dataset_name>/<YYYY-MM>/<arquivo>
```

Modelos treinados:

```text
models/trained/<model_name>/<YYYY-MM>/<artifact_name>
```

Manifestos de ultima versao:

```text
_catalog/datasets/raw/<dataset_name>/latest.json
_catalog/models/trained/<model_name>/latest.json
```

Essa organizacao deixa o bucket preparado para receber:

- datasets brutos;
- datasets tratados;
- modelos treinados;
- artefatos auxiliares de pipeline, inferencia e monitoramento.

### Exemplo para salvar modelo no MinIO

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

### Fluxo recomendado

1. subir o MinIO com `docker compose up -d`;
2. instalar o projeto com `pip install -e ".[notebooks]"`;
3. publicar o dataset com `credit-risk-ingest`;
4. ler dados no codigo com `load_credit_risk_dataset()`;
5. persistir modelos e artefatos no MinIO com `ModelRegistry`.

### Testes

Os testes unitarios validam:

- idempotencia do upload mensal do dataset;
- fallback local/Kaggle;
- manifestos `latest.json`;
- persistencia e leitura de modelos.

Execucao:

```powershell
python -m unittest discover -s tests\unit -v
```

## Observacoes

- Se o arquivo ja estiver disponivel em `data/credit_risk_dataset.csv`, voce pode publica-lo no MinIO e padronizar a leitura por la.
- A pasta `data/` pode continuar sendo usada como staging local, mas o ponto de leitura recomendado para pipeline passa a ser o MinIO.
- O script de ingestao grava por mes para facilitar historico, reprocessamento e reproducao de treino.
- O upload do dataset mensal e idempotente: se o arquivo ja existir no MinIO, ele nao e regravado.
- O `pyproject.toml` e agora a fonte principal de dependencias do pacote. `requirements.txt` foi mantido apenas como atalho para instalar `.[notebooks]`.
- Se o MinIO nao estiver no ar ou se o dataset ainda nao tiver sido publicado, os notebooks nao vao conseguir carregar a base.

## Proximos passos sugeridos

- validar tipos e valores ausentes;
- investigar distribuicoes e outliers;
- analisar balanceamento da variavel `loan_status`;
- preparar pipeline de treino para classificacao de risco.

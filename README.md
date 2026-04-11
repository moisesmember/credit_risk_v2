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

## Estrutura do projeto

```text
credit_risk_v2/
|-- data/
|   `-- credit_risk_dataset.csv
|-- notebook/
|   `-- estatistica.ipynb
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

### 2. Instalar as dependencias

```powershell
pip install -r requirements.txt
```

### 3. Iniciar o Jupyter

```powershell
jupyter notebook
```

Se preferir, voce tambem pode usar:

```powershell
jupyter lab
```

### 4. Abrir o notebook do projeto

No navegador, abra:

```text
notebook/estatistica.ipynb
```

### 5. Selecionar o kernel Python

Dentro do Jupyter, selecione o kernel do ambiente virtual criado para garantir que as dependencias instaladas em `requirements.txt` sejam utilizadas.

## Uso com Jupyter

Este projeto foi organizado para ser executado principalmente no **Jupyter Notebook**.

Fluxo esperado:

1. ativar o ambiente virtual;
2. iniciar o Jupyter;
3. abrir `notebook/estatistica.ipynb`;
4. executar as celulas em ordem.

O notebook atual utiliza `kagglehub` para baixar a base diretamente do Kaggle e tambem pode trabalhar com o arquivo local `data/credit_risk_dataset.csv`, quando ele ja estiver disponivel.

## Observacoes

- Se o arquivo ja estiver disponivel em `data/credit_risk_dataset.csv`, voce pode trabalhar localmente sem repetir o download.
- A pasta `data/` esta ignorada no versionamento por estar tratada como artefato local.

## Proximos passos sugeridos

- validar tipos e valores ausentes;
- investigar distribuicoes e outliers;
- analisar balanceamento da variavel `loan_status`;
- preparar pipeline de treino para classificacao de risco.

![Universidade Federal Rural do Rio de Janeiro](https://portal.ufrrj.br/wp-content/themes/portalufrrj/images/logomarca_ufrrj_cor.png)

---

### Programação Linear 2023-2
#### *Nova Iguaçu, 11 de novembro de 2023*
#### **Professor:** Erito Marques
#### **Alunos:** 
&emsp;Raphael Balmant

&emsp;Elis Braga

---

#### **Objetivo:** 

As rotas implementadas foram as seguintes no servidor de aplicação

- ``POST /heart/train``
- ``POST /heart/predict``

Exemplos de chamadas:

- Treinamento: ``curl --request POST --url http://localhost:8080/heart/train``
- Predição: ``curl --request POST \
  --url http://localhost:8080/heart/predict \
  --header 'Content-Type: application/json' \
  --data '{
	"age": 19,
   "sex": 0,
   "cp": 1,
   "trestbps": 120,
   "chol": 204,
   "fbs": 0,
   "restecg": 0,
   "thalach": 172,
   "exang": 0,
   "oldpeak": 1.4,
   "slope": 2,
   "ca": 0,
   "thal": 2
}
'``

---

#### **Como rodar:**

1) Instalar python >= 3.9 (https://www.python.org/)
2) Instalar poetry (https://python-poetry.org/)
3) cd code
4) poetry install
5) poetry shell

#### Depois de rodar os comandos acima, escolher uma das opções:

#### - Rodar como servidor de aplicação:

1) python .\src\main.py

#### - Rodar como aplicação standalone para validação do modelo:

2) python .\src\test.py

---

#### Acesso ao arquivo com o relatório

O relatorio pode ser encontrado na pasta: code\src\Relatorio.pdf

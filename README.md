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

O objetivo do presente trabalho é aplicar o método _Linear Programming Discriminant Analysis_ do artigo **LPDA: A new classification method based on linear programming** para obter um modelo de classificação binária através de um dataset previsamente selecionado (ver pasta data)

O artigo desenvolvido também gerou uma biblioteca em R que fornece métodos para utilizar o LPDA, cuja documentação encontra-se disponível no CRAN no endereço do [pacote lpda](https://cran.r-project.org/web/packages/lpda/index.html).

Transformar o código R para python ou tentar usar diretamente o R seria um desafio muito grande, devido a ter que conhecer particularidades tanto da linguagem R quanto das próprias estruturas de dados utilizadas pela biblioteca.

Ao invés disso, foi utlizada a biblioteca [Pulp](https://coin-or.github.io/pulp/), que possibilita realizar a descrição do problema de programação linear a ser resolvido de modo iterativo, e então dá a opção de utilização do solver da preferência do usuário. Assim como outras ferramentas disponíveis no mercado também fazem de maneira semelhante, como [AMPL](https://ampl.com/) por exemplo.

A partir da utilização da biblioteca em questão, foi possível resolver o problema de PL encontrando solução ótima, obtendo os coeficientes do plano H descrito no artigo, bem como a constante b. Além disso, foram implementados métodos para guardar esses valores em disco para reutilização posterior.

Guardando o arquivo de modelo em disco, foi possível implementar um servidor de aplicação que serve uma rota de predição, ficando assim o modelo fica disponível para utilização em um possível go live.

As rotas implementadas foram as seguintes no servidor de aplicação

- ``POST /train``
- ``POST /predict``

Exemplos de chamadas:

- Treinamento: ``curl --request POST --url http://localhost:8080/train``
- Predição: ``curl --request POST \
  --url http://localhost:8080/predict \
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
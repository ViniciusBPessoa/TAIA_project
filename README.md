# Classificação de Áreas de Conhecimento de Questões do ENEM utilizando computação evolutiva

## Grupo: Apolo Albuquerque, Irlan Andrade, Vinícius Pessoa

Beige and Gray Simplified Professional Portrait University Research Poster.png é nosso poster.

PaperMT.pdf é o relatório em formato de artigo.

### Análise Inicial da Base de Dados

- Analisamos os [microdados](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem) do ENEM e percebemos que as provas do ENEM digital, criado a partir do ENEM 2020, seguem a estrutura de uma questão por página em um PDF, o que pode facilitar nosso trabalho. Além disso, o ENEM de 2023 possui um arquivo .txt com todas as questões.
- Assim, teremos 185 questões por prova (5 a mais devido às questões de inglês e espanhol) e 4 provas cujas questões serão extraídas (ENEM 2020, 2021, 2022 e 2023). Pretendemos extrair e tratar os textos das questões dos PDFs com os conhecimentos já adquiridos na disciplina.
- Bibliotecas de Python podem nos ajudar a extrair o texto dos PDFs, organizar a estrutura das questões, remover stopwords, fazer embeddings, separar os dados de treino e teste, e então treinar um modelo para a classificação das questões em:
  
  - Ciências Humanas
  - Ciências da Natureza
  - Linguagens
  - Matemática
  
- No total, pretendemos extrair 740 questões divididas da seguinte forma: 180 de Ciências Humanas, Ciências da Natureza e Matemática, cada. Linguagens: 200 questões, resultando em uma base de dados relativamente balanceada.

### Exploração e Tratamento dos Dados

- A exploração dos dados foi feita utilizando a biblioteca PyPDF2, durante este processo foi encontradas algumas anomalias nos PDFs das provas de 2020 a 2022, como exemplos: algumas alternativas "A" estavam coladas aos enunciados (sem um /n), nas provas de 2021 e 2022 a área de conhecimento era lida, por exemplo, "Ciências da Natureza e suas T ecnologias".
  
- A melhor alternativa encontrada de importação dos textos dos PDFs foi utilizando um regex para splitar os blocos de questão que sempre iniciavam com "Questão - XXX..." e, dado que a maioria das questões continha suas alternativas em uma linha, as ultimas 5 linhas de cada bloco foram consideradas como sendo as alternativas. Inicialmente duas bases dados foram criadas: uma com os enunciados e alternativas separadas e outra com o texto deles juntos em uma só coluna.
  
- A prova de 2023 tinha um arquivo .txt com suas questões, o que, relativamente, facilitou o trabalho. Para o caso dessa prova, as questões sempre inciavam com "QUESTÃO XXX".
  
- Tendo em vista que as provas de Linguagens (Inglês e Espanhol) vieram em PDFs separados, foi-se importadas questões duplicadas, o que foi resolvido.

- Células vazias são possíveis imagens, foi adicionado "possível imagem" como padrão para partes sem preenchimento nos CSVs. Isso também ajuda na parte de pré-processamento de texto.

- O pré-processamento foi feito aplicando algumas como deixar tudo minúsculo, remover pontuações, fazer tokenização, fazer Stemming, remover stopwords e palavras pequenas mantendo termos importantes. Também foi considerado que algumas alternativas de matemática podem contem somente um dígito (número).

- No total temos 4 base de dados no caminho `Exploração e Tratamento dos Dados/CSV`, todas com 753 elementos:

  - `alternativas_separadas.csv`: uma base **sem** pré-processamento contendo as colunas 'Ano', 'Enunciado', 'Alternativa_A', 'Alternativa_B', 'Alternativa_C', 'Alternativa_D', 'Alternativa_E' e 'Area_de_Conhecimento';
  - `tudo_junto.csv`: uma base **sem** pré-processamento contendo as colunas 'Ano', 'Enunciado_Alternativas' e 'Area_de_Conhecimento';
  - `alternativas_separadas_pp.csv`: uma base **com** pré-processamento contendo as colunas 'Ano', 'Enunciado', 'Alternativa_A', 'Alternativa_B', 'Alternativa_C', 'Alternativa_D', 'Alternativa_E' e 'Area_de_Conhecimento';
  - `tudo_junto_pp.csv`: uma base **com** pré-processamento contendo as colunas 'Ano', 'Enunciado_Alternativas' e 'Area_de_Conhecimento';

  A grande diferença entre as bases `alternativas_separadas` e `tudo_junto`, para além da junção de enunciado e alternativas, é o processo no qual foi adicionado "possível imagem" nas células vazias das alternativas da base `alternativas_separadas`.
  
- Na pasta `Exploração e Tratamento dos Dados` estão contidos os diretórios `CSV`: que aloca as bases dados já construídas, `Provas`: que contém os arquivos PDF e txt das provas e `data.ipynb`: código Python onde foram realizados os experimentos e tratamentos dos dados.

### Exploração a Análise dos Dados

- É necessário verificar como esta realizada a distribuição de questões dentro de nossa base de dados, tendo em vista que caso a mesma estive-se desbalanceada mais tratamentos deveriam ser feitos, a distribuição final foi de (**27.40%** em Linguagens, **22.73%** em Ciências Humanas, **24.24%** em Ciências da Natureza, **25,63%** em Matemática), logo satisfatoriamente **balanceada**.
- Outro ponto que necessita ser abordado é o de como se comportam as falhas dentro do dataste, e conde as mesmas se agrupam (Para uma visão mais bem elaborada quanto a esse ponto recomendo a leitura do relatório ao fim desse documento), onde é notável a ausência de falhas nos enunciados mas a dominância dos mesmos quase que exclusivamente em matemática e natureza, essa descoberta nus leva a buscar ate conde interferem essas em comparação com a seleção e utilização unicamente do enunciado(que esta completo).
 - Tudo isso pode ser encontrado na pasta `Analitcs` no arquivo `analise_dataset.ipynb`.

### Modelos Feijão com Arroz

- Com a intenção de comparar com algoritmos de Aprendizagem por Reforço, foram treinados 3 modelos clássicos de classificação utilizando TF-IDF para vetorização que incluem:
  
  - **Naive Bayes**: Foi utilizado o modelo `MultinomialNB` em sua forma padrão, entretanto, foi adicionado o parâmetro `sublinear_tf=True` no TfidfVectorizer que o deixa mais sensível a termos menos frequentes mas potencialmente mais significativos, que, no geral, melhorou sua acurácia nos datasets;
  - **Logistic Regression**: Foi utilizado o modelo `LogisticRegression` com os parâmetros `multi_class='multinomial'` e `solver='lbfgs'`, o `solver='lbfgs'` especifica que o modelo de regressão logística usará o algoritmo L-BFGS para otimizar os coeficientes. Aqui também incluí o parâmetro `sublinear_tf=True` no TfidfVectorizer;
  - **Random Forest**: Foi utilizado o modelo `RandomForestClassifier` com os parâmetros `n_estimators=200` e `random_state=3`, `n_estimators=200` determina a quantidade de árvores na floresta e a quantidade 200 foi definida após alguns testes: 100 é a quantidade padrão, 200 foi o máximo encontrado para a melhor acurácia do modelo. Mais de 200 demandam mais poder computacional, refinam o modelo mas não melhoram a acurácia.

  Todos os modelos foram treinados com um test split de 80/20 (80% para treino e 20% para teste).

- Abaixo segue tabela comparativa com as acurácias para cada dataset de cada modelo e a média das acurácias de cada dataset:
  
  | Dataset                 |   Naive Bayes |   Logistic Regression |   Random Forest |   Média de Acurácia |
  |:---|---:|---:|---:|---:|
  | alternativas_separadas   |      0.847682 |              0.860927 |        0.781457 |            0.830022 |
  | alternativas_separadas_pp|      0.874172 |              0.894040 |        0.860927 |            0.876380 |
  | tudo_junto               |      0.854305 |              0.854305 |        0.761589 |            0.823400 |
  | tudo_junto_pp            |      0.894040 |              0.894040 |        0.827815 |            0.871965 |

  De imediato, pode-se observar que as bases de dados pré-processadas tiveram acurácias melhores das que não tiveram esse processo.

- Na pasta `Modelos Feijão com Arroz` estão contidos os diretórios `CSV`: que aloca as bases dados e `main.ipynb`: código Python onde foram realizados os treinamentos dos modelos.

### Metodologia de Utilização do Google Gemini

- Utilização do Gemini foi feita como uma base comparativa para verificar a eficácia de modelos generativos no mercado em comparação com os clássicos na atividade de classificação textual.
- Inicialmente foi gerada uma chave para acesso a API (Aplication Programing Interface), chave essa que esta contida em `Analitcs` e `.env`.
- Com a chave em mão inicializamos a engenharia de  prompt  para  para realizarmos as requisições solicitando as respostas.
- Para esse experimento utilizamos o modelo `gemini 1.5-flash` dado o seu custo beneficio.
- O código que realiza a requisição pode ser encontrado em `Analitcs` e `gemini_data.ipynb` e para carregar a base de dados que pretende enviar ao gemini basta alterar o a linha de código  *data = pd.read_csv(r'CSV\alternativas_separadas.csv')* para garantir que o nome do arquivo é o desejado.
- Todas as respostas fornecidas pelo modelo foram realizadas no sequente padrão como foi solicitado no prompt '0 \n'.  onde de 0 a 3 representam nossas possíveis classes além de necessitar de conversão para a forma inteira previamente a analise.

### Analise de dados do Google Gemini

- Já dentro de `Analitcs` e `gemini_analitcs.ipynb` é possível observar como ficaram distribuídas as respostas do google gemini em acurácia geral, por área  e matriz de difusão.
- Utilizando a biblioteca *pickle* salvei as listas já convertidas para facilitar a analise. (Caso queira acessar tanto os dados de unicamente o enunciado quanto enunciado e alternativas basta  alterar o nome de **with open('lista_1a1.pkl', 'rb')** as file: contendo tudo junto para **with open('lista_enum_1a1.pkl', 'rb')** as file.
- A analise gráfica recomendo a leitura do relatório escrito ao fim desse documento, mesmo assim no fim fica nítida a eficiência geral do modelo 81%, como também através da matriz de confusão é notável a confusão realizada pelo modelo ao classificar varias vezes uma pergunta de linguagens como ciências humanas.

### Analise de dados dos modelos de redes neurais

- Utilizamos primeiramente **Aprendizagem Supervisionada**:
	- O treinamento supervisionado de redes neurais é basicamente ensinar uma rede com exemplos onde você já sabe a resposta. Você dá à rede dados com as respostas certas e ela aprende a ajustar seus parâmetros para que, quando vê novos dados, possa prever a resposta correta. Tentando sempre minimizar o erro entre o que a rede prevê e o que deveria prever, e isso é feito ajustando os parâmetros da rede usando algoritmos específicos.
	- 
- **Redes Neurais Densas**: Estas redes possuem camadas onde cada neurônio está conectado a todos os neurônios da camada anterior. São amplamente usadas para tarefas de classificação e regressão. No treinamento supervisionado, você fornece dados rotulados para a rede aprender a mapear entradas para saídas desejadas.
    
- **Redes Neurais Recorrentes (RNNs)**: São projetadas para lidar com dados sequenciais, como séries temporais ou texto. Elas têm conexões que permitem a propagação de informações ao longo do tempo. No treinamento supervisionado, você fornece sequências de dados e os rótulos associados para que a rede aprenda padrões temporais e contextuais.

Toda essa analise gráfica pode ser encontrada em `Analitcs` e `Modelos_maiores.ipynb`.

### Rascunho do Relatório em Formato de Artigo

Link no Overleaf: https://www.overleaf.com/read/frvhftjgchkr#6b841c

# Guia Completo de Preparação para a Apresentação do Trabalho de MPI

Este documento foi elaborado para ser o seu guia definitivo de estudos para a apresentação do trabalho da disciplina de Programação Paralela/Sistemas Distribuídos focada em MPI (Message Passing Interface). 

O objetivo deste material é fornecer, de forma **didática, detalhada e exaustiva**, toda a teoria necessária (baseada nos slides fornecidos), a explicação profunda da implementação do código e as respostas para as reflexões exigidas pela professora, além de uma seção robusta de "Possíveis Perguntas e Respostas" para que você não seja pego de surpresa na apresentação.

Leia este documento com atenção, do início ao fim.

---

## ÍNDICE

1. [PARTE 1: Teoria Completa sobre MPI e Processamento Paralelo](#parte-1-teoria-completa-sobre-mpi-e-processamento-paralelo)
2. [PARTE 2: Processamento de Imagens e Filtros de Suavização](#parte-2-processamento-de-imagens-e-filtros-de-suavização)
3. [PARTE 3: O Trabalho e Seus Requisitos](#parte-3-o-trabalho-e-seus-requisitos)
4. [PARTE 4: Estrutura do Código e Funcionamento Detalhado](#parte-4-estrutura-do-código-e-funcionamento-detalhado)
5. [PARTE 5: Respostas Oficiais para as Reflexões Pedidas](#parte-5-respostas-oficiais-para-as-reflexões-pedidas)
6. [PARTE 6: Possíveis Perguntas da Professora na Apresentação](#parte-6-possíveis-perguntas-da-professora-na-apresentação)

---

<br>
<br>
<br>

## PARTE 1: Teoria Completa sobre MPI e Processamento Paralelo

Nesta seção, vamos cobrir detalhadamente todos os conceitos teóricos dos slides `spd4_comunicação_04 - MPI.pdf` e `spd4_comunicação_04 - MPI-2.pdf`. Se a professora perguntar qualquer conceito básico, a resposta estará aqui.

### 1.1 O que é Computação de Alto Desempenho e Cluster Computing?

A computação de alto desempenho (HPC) visa resolver problemas complexos que exigiriam dias ou meses para serem resolvidos em um computador comum. Para isso, utilizamos arquiteturas que permitem a execução de múltiplas instruções simultaneamente.

Um **Cluster Computing** é um tipo de sistema computacional formado por um aglomerado de nós (vários computadores).
Esses computadores são:
- Geralmente idênticos em hardware.
- Executam o mesmo sistema operacional.
- Estão conectados em uma rede local de alta velocidade.
- O usuário e as aplicações enxergam esse cluster como uma **visão única do sistema** (como se fosse um supercomputador gigantesco).

**Exemplo Prático Nacional:** 
Nos slides, é citado o **SINAPAD** (Sistema Nacional de Processamento de Alto Desempenho) e o **Supercomputador Santos Dumont** no LNCC (Laboratório Nacional de Computação Científica). O Santos Dumont é o maior supercomputador da América Latina voltado para pesquisa pública, e ele nada mais é do que um gigantesco cluster com milhares de nós interconectados.

### 1.2 Os Desafios da Programação Paralela

Quando você programa de forma sequencial, você tem um fluxo único. Quando vamos para um ambiente de cluster, surgem vários desafios para o programador:
1. **Distribuição de Trabalho:** O computador não adivinha como dividir a tarefa. O *programador* é o único responsável por dividir o problema maior em partes menores que possam ser calculadas paralelamente.
2. **Comunicação:** Os nós do cluster precisam trocar informações (ex: enviar a imagem original e receber a imagem processada).
3. **Coordenação e Sincronização:** Os processos precisam saber a hora de esperar (ex: ninguém calcula o speedup até que todos terminem de processar a imagem).

É para resolver o desafio da **Comunicação, Coordenação e Sincronização** que utilizamos o MPI.

### 1.3 O que é o MPI?

**Atenção:** Isso é "pegadinha" clássica de prova/apresentação.
O **MPI (Message Passing Interface)** **NÃO** é uma biblioteca. Ele é uma **ESPECIFICAÇÃO DE INTERFACE** (um padrão).
Isso significa que o MPI é apenas um documento com regras dizendo: "Para enviar uma mensagem, a função deve se chamar X e receber os parâmetros Y e Z".

As pessoas pegam esse documento (essa especificação) e criam as bibliotecas que realmente fazem o trabalho. Exemplos de implementações do padrão MPI:
- MPICH
- Open MPI (muito comum em Linux)
- MS-MPI (da Microsoft, para Windows)
- Intel MPI

### 1.4 A Arquitetura de Memória Distribuída

Em threads convencionais, a memória é compartilhada. Se a thread A altera uma variável, a thread B enxerga na hora.

No MPI (desenhado para clusters), a arquitetura é de **Memória Distribuída**.
Isso significa que o Processo 0 (que pode estar no Computador A) tem a sua própria memória RAM, e o Processo 1 (no Computador B) tem a sua própria RAM.
**Eles não conseguem ler a variável um do outro diretamente!**

Se o Processo 0 tem uma variável `x`, e o Processo 1 precisa do valor de `x`, o Processo 0 tem que empacotar o valor de `x` em uma **mensagem** e enviar pela rede (ou pelo barramento interno) até a memória do Processo 1. Daí o nome: Message Passing (Troca de Mensagens).

### 1.5 Nomenclaturas Iniciais Importantes

Para programar em MPI, você precisa dominar o vocabulário básico:

- **Comunicador (Communicator):** É o identificador que define um grupo de processos. O comunicador padrão que engloba TODOS os processos da execução atual é chamado de `MPI_COMM_WORLD` (o mundo todo).
- **Grupo:** É o conjunto de processos pertencentes a um comunicador. 
- **Rank:** É a **identificação única (ID)** de cada processo dentro de um grupo.
  - Ele começa sempre em `0`.
  - Se você executar seu programa com 4 processos, os ranks serão `0, 1, 2 e 3`.
  - O Rank é usado como o "endereço". Quando você envia uma mensagem, você diz: "Enviar para o rank 2".
- **Size:** É o tamanho do grupo (quantos processos existem). No exemplo acima, o size é `4`.

### 1.6 Comunicação Ponto-a-Ponto

A comunicação ponto-a-ponto ocorre estritamente entre DOIS processos (um envia, o outro recebe).
As operações básicas são:
- **Send:** Envia a mensagem.
- **Recv:** Recebe a mensagem.

Existem duas formas principais de fazer isso:
1. **Bloqueante (`Send` / `Recv`):** A execução do programa para (fica bloqueada) até que a mensagem seja completamente enviada do buffer local e possa ser modificada em segurança, ou até que a mensagem seja totalmente recebida.
2. **Não-Bloqueante (`isend` / `irecv`):** O processo manda enviar e já continua executando as próximas linhas de código (assíncrono). Ele precisa checar depois se a operação concluiu antes de tentar usar a variável.

*(Nota: Nosso trabalho de imagem **não** usa ponto-a-ponto explicitamente para as fatias, usamos operações coletivas que são mais eficientes).*

### 1.7 Operações Coletivas

Operações coletivas envolvem **TODOS** os processos de um comunicador. Todo mundo tem que chamar a função, senão o programa trava (deadlock). Elas são otimizadas para alto desempenho.

As principais vistas nos slides:
- **Broadcast (`Bcast`):** Um processo (geralmente o rank 0, chamado de *root*) tem um dado e envia uma cópia para TODOS os outros processos do grupo.
- **Scatter:** Um processo (root) tem um vetor grande. Ele **divide** (espalha) o vetor em pedaços iguais e manda o pedaço 1 para o rank 0, o pedaço 2 para o rank 1, etc.
- **Gather:** O oposto do Scatter. Todos os processos têm um pequeno pedaço de dados. O root **coleta** esses pedaços e os junta num vetor grande só para ele. *(No nosso trabalho, usamos a variação `Gatherv`, que permite receber pedaços de tamanhos diferentes, já que a imagem pode não ser divisível igualmente por todos os processos).*
- **Reduce:** Todos os processos têm um valor. O root coleta esses valores aplicando uma operação neles, como SOMA (`MPI.SUM`), MAX, MIN, etc.

### 1.8 A Regra de Ouro do mpi4py (Crucial para o Trabalho!)

Se você usar Python e a biblioteca `mpi4py`, existe uma diferença brutal entre usar métodos com letra **minúscula** e com letra **Maiúscula**. Isso não é estilo de código (PEP8), é arquitetura interna.

**Métodos com Minúscula (ex: `comm.bcast`, `comm.send`):**
- Foram feitos para objetos Python genéricos (listas, dicionários, classes).
- Como o C (linguagem por trás do MPI) não entende o que é um dicionário Python, a biblioteca tem que **serializar** o objeto. Ela usa a biblioteca `pickle` do Python, transforma o dicionário numa tripa de bytes (string binária), envia pela rede, e o destinatário faz o caminho inverso (unpickle).
- **Problema:** Isso gera "muitas camadas extras" de processamento. É incrivelmente LENTO. Se você passar uma imagem (matriz de milhões de pixels) por `comm.bcast`, vai demorar muito mais do que o processamento da imagem em si.

**Métodos com Maiúscula (ex: `comm.Bcast`, `comm.Send`):**
- Foram feitos para trabalhar com **buffers contíguos de memória em C**.
- Em Python, quem fornece esse tipo de buffer é a biblioteca **NumPy** (que é escrita em C sob os panos).
- Quando você usa `comm.Bcast(buffer_numpy)`, o mpi4py não serializa nada. Ele pega o ponteiro de memória onde o array numpy começa na RAM e manda a placa de rede jogar aqueles bytes puros diretamente para o destino.
- **Resultado:** Desempenho equivalente à linguagem C pura. Extremamente rápido!

*Essa é a explicação do porquê o requisito do trabalho exige "operações com maiúscula (buffer numpy)".*

<br>
<br>
<br>

---

## PARTE 2: Processamento de Imagens e Filtros de Suavização

### 2.1 O que é uma Imagem Digital?

Para o computador, uma imagem não tem "cor" mágica. Ela é uma gigantesca matriz (ou grade) de números.
- Em imagens em escala de cinza (que usamos no trabalho), cada ponto (pixel) é apenas um número inteiro entre 0 e 255.
- `0` significa preto total (ausência de luz).
- `255` significa branco total (intensidade máxima de luz).
- Valores intermediários (ex: 128) são tons de cinza.

### 2.2 Problemas: Ruídos e Bordas Abruptas

Quando uma foto é tirada com pouca luz (ISO alto da câmera), ou sofre interferência na transmissão digital, ela adquire "ruído" (noise).
Existem vários tipos:
- **Ruído Gaussiano:** Variações aleatórias na iluminação. Parece a "estática" de TV velha.
- **Ruído Impulsivo (Sal e Pimenta):** Pixels aleatórios ficam completamente pretos ou completamente brancos.

**Para que servem os filtros de suavização?** (Slide 4 da aula 2)
Servem para reduzir o nível de troca de contraste entre pixels próximos. 
Eles são usados para:
- Corrigir serrilhados.
- Suavizar transições abruptas.
- Remover ruídos.

### 2.3 O Filtro de Média (Convolução)

Como o filtro de média funciona?
Ele pega uma "janela" 3x3 de pixels (um quadrado de 3 pixels de largura por 3 de altura).
Ele posiciona o centro dessa janela em cima de um pixel que queremos processar.
Então, ele pega o valor dos 9 pixels que estão dentro dessa janela, soma tudo, divide por 9 (tira a média aritmética) e substitui o pixel central pelo resultado.

Isso é feito para **cada um** dos pixels da imagem. Esse processo de deslizar uma janela sobre uma matriz aplicando uma operação matemática é chamado de **Convolução**.

**Efeito:** Ele mistura o pixel com seus vizinhos. Isso causa um efeito de "borrão" (blur). O ruído gaussiano é dissolvido, mas a imagem perde nitidez. O filtro de média é péssimo para ruído sal e pimenta, pois um pixel branco extremo (255) vai puxar a média de toda a vizinhança para cima.

### 2.4 O Filtro de Mediana (Sugerido no Trabalho)

Em vez de somar tudo e dividir por 9, o filtro de mediana pega os 9 valores da janela 3x3 e os **ordena** do menor para o maior.
Exemplo dos valores: `[10, 12, 15, 15, 20, 25, 30, 255, 255]`.
Ele pega o valor exatamente no meio da lista ordenada (no caso, o 5º valor, que é `20`).

**Por que a Mediana é genial contra ruído Sal e Pimenta?**
Porque os pixels defeituosos (0 ou 255) sempre vão parar nas extremidades da lista ordenada. O centro da lista (a mediana) sempre será um pixel "saudável" e representativo daquela vizinhança. Assim, a mediana limpa a imagem sem borrar as bordas com a mesma severidade da média.

**Custo Computacional:** Ordenar 9 elementos para cada pixel de uma imagem de milhões de pixels exige muito da CPU. O filtro de Mediana é muito mais "pesado" que o de Média. Daí a necessidade de paralelizar!

<br>
<br>
<br>

---

## PARTE 3: O Trabalho e Seus Requisitos

O enunciado da professora foi muito claro e cheio de detalhes técnicos. Vamos dissecá-lo.

**A Ideia Central:**
Comparar filtros de imagem utilizando o MPI.

**O que o código deveria fazer (e FAZ):**
1. **Dois Filtros:** O código implementa o Filtro de Média 3x3 e o Filtro de Mediana 3x3.
2. **Mesma Imagem:** O código carrega a imagem "entrada.jpg" em escala de cinza e aplica ambos os filtros nela. Se não achar, cria uma gigantesca (4000x4000) de forma sintética com muito ruído, justamente para forçar a máquina a suar (se a imagem for pequena, o MPI mais atrapalha que ajuda).
3. **Versão Sequencial e Paralela:** O código roda a baseline sequencial e depois a versão distribuída com MPI.
4. **Buffer Numpy:** O código usa `comm.Bcast` e `comm.Gatherv` (maiúsculas) transmitindo arrays numpy puros, alcançando o máximo de performance que o Python permite no MPI.
5. **Cálculo de Speedup, Variância e Outliers (Benchmark):** Nós implementamos a regra matemática do IQR (Interquartile Range) para jogar fora as execuções fora do padrão (outliers). Calculamos o tempo de cada iteração, variância e finalmente o Speedup (Tempo Sequencial / Tempo Paralelo).
6. **Início Frio (Overhead Simétrico):** A professora especificou explicitamente: *"o testador não deve iniciar todo o ambiente MPI via comando mpiexec para não forçar um início frio do teste... o overhead de inicialização é simétrico... só adicionaria tempo extra"*. 
   Isso significa que não podíamos fazer um script shell rodando `mpiexec python main.py` 30 vezes. O `mpiexec` deveria ser rodado 1 única vez, e dentro do nosso código Python, deveríamos fazer um loop de `for it in range(30)`. Foi exatamente assim que o `main.py` foi construído!
7. **Tabela e Gráfico:** O código gera automaticamente os gráficos (`tempo_vs_processos.png`, `speedup_vs_processos.png`) e um relatório em texto e CSV na pasta `output/`.

<br>
<br>
<br>

---

## PARTE 4: Estrutura do Código e Funcionamento Detalhado

O projeto está dividido em três arquivos para organização e modularidade profissional:

### 1. `filters.py` (O coração matemático)
Este arquivo contém a lógica pura de como a média e a mediana são calculadas, sem saber nada sobre MPI.
Temos 4 funções principais nele:
- `apply_mean_filter_fast` / `apply_median_filter_fast`: São usadas para medir o tempo sequencial (baseline). Foram escritas usando "numpy slicing" e vetorização. Elas são muito mais rápidas que fazer loops `for i` `for j` no Python. São as nossas referências de "o quão rápido 1 processador consegue ir no seu melhor cenário".
- `compute_mean_block` / `compute_median_block`: Estas são as estrelas do mundo paralelo. Quando fatiamos a imagem, cada processo MPI chama essa função passando apenas o seu pedaço de imagem (bloco). O diferencial genial destas funções são as **linhas de halo**. (Explicaremos isso nas respostas abaixo).

### 2. `benchmark.py` (O analista de dados)
A professora exigiu cálculo de estatísticas. Este arquivo é uma mini-biblioteca estatística.
- `remove_outliers_iqr`: Implementa o IQR (Intervalo Interquartil).
  - Pega todos os tempos, acha o Q1 (25%) e Q3 (75%). A diferença (Q3 - Q1) é o IQR.
  - Qualquer medição que foi muito rápida ou muito lenta (que escapou de `Q1 - 1.5*IQR` até `Q3 + 1.5*IQR`) é um outlier (talvez o Windows Update começou a rodar na hora, ou o antivírus ativou). Essa função expurga essa sujeira estatística.
- `compute_stats`: Calcula a Média, Variância (usando `np.var`) e Desvio Padrão.
- As outras funções (`plot_speedup`, `gerar_relatorio`, etc.) geram as saídas na pasta `output`.

### 3. `main.py` (O orquestrador MPI)
Este é o arquivo que você chama no terminal com o comando:
`mpiexec -n <numero_processos> python main.py`

**O fluxo passo-a-passo dentro dele é:**
1. **Inicialização do MPI:** Pega o comunicador, descobre quem ele é (`rank`) e quantos são (`size`).
2. **Apenas o Rank 0 (O Mestre):** Lê a imagem do disco (ou gera a sintética).
3. **Distribuição do Tamanho (Shape):** O Rank 0 usa o `comm.bcast` (minúsculo mesmo) apenas para mandar 2 inteiros (altura e largura). Mandar 2 inteiros com minúscula é irrisório, não causa gargalo, e avisa todo mundo de qual tamanho de array eles precisam preparar.
4. **Alocação de Memória:** Os Ranks 1, 2, 3... alocam um array vazio gigante de tamanho HxW (`np.empty`).
5. **O GRANDE BCAST (`comm.Bcast`):** O Rank 0 manda a imagem verdadeira (milhões de bytes) para todos de uma só vez, usando o buffer numpy (letra Maiúscula). Agora todo mundo na rede tem uma cópia da imagem!
6. **Benchmark Sequencial:** O Rank 0 roda as funções sequenciais para ter o tempo base.
7. **Divisão (Fatias):** O algoritmo `calcular_fatias` diz para cada rank em qual linha ele começa e em qual termina. (ex: H=4000. Proc 0 faz de 0 a 1000, proc 1 de 1000 a 2000...).
8. **Loop Paralelo (30 vezes):** 
   - `comm.Barrier()`: Segura todos os processos na mesma linha. Dispara o cronômetro local.
   - Cada rank recorta da imagem (que ele recebeu no Bcast) apenas o seu pedaço (junto com 1 linha a mais acima e abaixo, os *halos*).
   - Roda a função do filtro (`compute_mean_block` ou `compute_median_block`). O resultado é devolvido achatado (1D).
   - `comm.Gatherv`: O Rank 0 coleta os resultados de todo mundo. Como o resultado foi "achatado", ele entra de forma contígua e perfeitamente encaixada no buffer gigante de saída do Rank 0, numa eficiência incrível.
   - `comm.Barrier()`: Segura todo mundo de novo. Para o cronômetro.
   - Calcula o máximo tempo entre todos (a iteração só acaba quando o último processo terminar).
9. O Rank 0 salva a imagem final cortada de volta pra 2D.
10. O Rank 0 atualiza um arquivo `historico.json`. Se você rodar com `-n 2`, ele grava. Quando rodar com `-n 4`, ele abre o json, junta os resultados do 2 e do 4, gera a tabela de comparação atualizada e o gráfico. Tudo automático!

<br>
<br>
<br>

---

## PARTE 5: Respostas Oficiais para as Reflexões Pedidas

A professora exigiu respostas para quatro perguntas reflexivas. Você as encontrará também no arquivo gerado automaticamente dentro da pasta `output/relatorio.txt` após rodar o programa, mas a base e a explicação completa estão aqui para você saber explicar de cabeça:

### Reflexão 1: O filtro de mediana paralelo teve speedup maior ou menor que o de média? Por quê?

**Resposta:** O filtro de mediana tem, invariavelmente, um **Speedup MAIOR** que o de média em processamento paralelo MPI.

**A Explicação (Por quê?):** 
Em processamento paralelo, nós temos uma balança: **Custo de Processamento vs Overhead de Comunicação**.
- O envio e coleta de fatias via rede/MPI leva tempo. Isso é o overhead de comunicação.
- O filtro de Média é muito "leve" para a CPU (apenas 9 somas e 1 divisão). Quando dividimos essa tarefa entre vários processos, o tempo de CPU salvo quase não compensa o tempo gasto sincronizando e enviando os dados (comunicação). O speedup não decola de forma linear.
- Já o filtro de Mediana é muito "pesado" (exige ordenar 9 valores para cada pixel, o que demanda acessos à memória cache complexos). Como a tarefa em si demora muito na CPU, ao dividi-la por 4 processos, a economia de tempo é colossal, diluindo e superando em muito o custo de comunicação MPI. Logo, a "recompensa" pela paralelização (o speedup) é bem mais alta.

### Reflexão 2: O que acontece com os pixels nas fronteiras entre processos? O grupo tratou isso? Como?

**Resposta:** Sim, tratamos com **"Linhas de Halo" (ou Ghost cells)**.

**A Explicação:**
Se nós cortarmos a imagem estritamente em fatias e mandarmos cada processo calcular a sua, teremos um grave defeito nas bordas da fatia. 
Para calcular o pixel da PRIMEIRA linha de uma fatia de um processo, a máscara 3x3 do filtro precisa "olhar" para a linha de cima. Mas a linha de cima ficou com o processo vizinho! Se o processo assumir que a linha de cima não existe (borda preta), vai aparecer uma fita (risco) estranha no meio da imagem juntada final.

**Como resolvemos:**
O nosso código (`benchmark_paralelo_filtro` no `main.py`) extrai o que chamamos de **"Halo"**.
Se o processo 1 é responsável pelas linhas de 1000 a 1999:
1. Ele recorta localmente as linhas de **999 a 2000**.
2. A linha 999 (halo_top) e a linha 2000 (halo_bottom) não pertencem a ele, mas ele as puxa **somente para leitura**, como vizinhança.
3. O filtro é aplicado, calculando os resultados corretamente para as linhas 1000 a 1999 (porque agora elas têm os vizinhos presentes).
4. No final (dentro de `compute_mean_block`), as linhas de halo (a primeira e a última do bloco modificado) são **descartadas**.
5. O processo envia de volta no `Gatherv` apenas os resultados da sua área legítima (1000 a 1999) perfeitamente calculados.
*(Obs: Como fizemos `comm.Bcast` da imagem inteira para todo mundo logo no início da execução, a imagem toda está na memória de todos os nós. Puxar essa 1 linha de halo é uma simples cópia local de memória, sem necessidade de troca de mensagens ponto-a-ponto entre os nós!)*

### Reflexão 3: Por que foi usado comm.Bcast e não comm.bcast para enviar a imagem? O que mudaria no desempenho?

**Resposta:** Essa é a "Regra de Ouro" do MPI com Python (Mencionada no slide 15 da aula 2). Usamos `Bcast` (maiúscula) porque ele envia buffers puros C (numpy) garantindo altíssima velocidade.

**A Explicação:**
- `comm.bcast` (minúscula) foi feito para enviar objetos genéricos em Python (listas, dicionários, objetos). Para funcionar em um ambiente C (que o MPI usa), o Python tem que serializar todo o objeto numa string de bytes usando o pacote `pickle`. Isso é chamado de overhead de serialização e desserialização. Para um array de imagem enorme, isso geraria um congelamento/lentidão massiva.
- `comm.Bcast` (Maiúscula) permite enviar o ponteiro de memória exato da nossa matriz Numpy. O mpi4py diz à placa de rede: "pegue esses blocos de memória e mova para lá", não há passo de serialização no meio, a comunicação acontece quase na velocidade nativa do hardware. 
**O que mudaria?** Se usássemos minúscula (`comm.bcast`), o tempo gasto na comunicação afogaria o benchmark. O tempo total do paralelo ficaria pior que o sequencial e o speedup seria pífio (fracionário, ex: 0.2x).

### Reflexão 4: Se o número de processos não divide exatamente a altura da imagem, o que o código faz?

**Resposta:** O código distribui o "resto da divisão" equilibradamente, dando 1 linha extra para os primeiros processos.

**A Explicação:**
Se a imagem tem altura de 4000 pixels e executamos em 3 processos, 4000 não é divisível por 3 (4000 // 3 = 1333, com resto 1).
Se cada um pegasse 1333 linhas, ficaríamos com 1 linha faltando no final da imagem (imagem incompleta).
A função `calcular_fatias` no `main.py` resolve isso de forma elegante:
```python
base  = height // n_proc     # Base = 1333
resto = height %  n_proc     # Resto = 1
```
Aí, fazemos um loop por cada processo e distribuímos o resto: se o rank do processo for menor que o resto, ele ganha uma linha extra.
- Processo 0 (rank 0 < 1): Recebe base + 1 = 1334 linhas.
- Processo 1 (rank 1 não é < 1): Recebe base = 1333 linhas.
- Processo 2 (rank 2 não é < 1): Recebe base = 1333 linhas.
1334 + 1333 + 1333 = 4000 exatos. Todo trabalho foi distribuído e ninguém faz muito mais esforço que o outro.

<br>
<br>
<br>

---

## PARTE 6: Possíveis Perguntas da Professora na Apresentação

Para apresentar com maestria, preparamos um "tira-dúvidas" com perguntas cabeludas que a professora pode fazer para testar se vocês realmente entenderam o código e a teoria.

### ❓ 1. Professora: "Por que vocês usaram Gatherv em vez de apenas Gather?"
**R:** "Porque as alturas das fatias processadas podem ser diferentes, professora. Como respondido na nossa reflexão, se a altura da imagem não for divisível pelo número de processos, um processo pode retornar 1001 linhas, e outro 1000 linhas. O `Gather` comum exige que todo mundo devolva exatamente o mesmo tamanho (mesma contagem de elementos). Já o `Gatherv` (Gather Variable) aceita tamanhos diferentes, passando os vetores `send_counts` (tamanho que cada um manda) e `displacements` (onde cada pedaço começa no array final)."

### ❓ 2. Professora: "Vocês usam Barreira (`comm.Barrier()`). Qual a função exata disso ali no loop de benchmark?"
**R:** "No slide 16 da aula 2, o uso de `Barrier` é indicado para medir o tempo real na rede e garantir o isolamento. Como queremos calcular o tempo de execução e há um cronômetro envolvido, precisamos garantir que ninguém comece com vantagem ou encerre antes dos outros. O primeiro `Barrier` garante que todos os nós larguem ao mesmo tempo. O segundo `Barrier` no fim do loop segura o cronômetro, de forma que o tempo contabilizado seja o tempo em que *a tarefa total* levou para terminar em todo o cluster, alinhando a execução."

### ❓ 3. Professora: "Por que vocês rodaram 30 repetições dentro do código Python em vez de fazer um script Bash mandando 30 mpiexec por fora?"
**R:** "Por recomendação do próprio enunciado. Se fizéssemos `mpiexec python main.py` num script bash 30 vezes, cada execução do bash criaria uma nova topologia MPI, alocaria os sockets de rede e iniciaria as VMs do Python do zero. Isso adicionaria um "início frio" (cold start) gigantesco ao benchmark que não tem relação com a matemática do filtro. Executando o loop de 30 interno, nós iniciamos o ambiente MPI uma única vez, a conexão já está aquecida, fazemos um *warm-up* fora do tempo para aquecer os caches de CPU, e medimos apenas as execuções puras dos filtros."

### ❓ 4. Professora: "No `Gatherv`, reparei que vocês transformam o resultado do filtro em um array 1D (`.flatten()`). Por que não devolveram o array 2D original da imagem?"
**R:** "Isso foi uma otimização técnica profunda para uso com buffers numpy do `mpi4py`. O MPI (no seu baixo nível em C) entende a memória como um array contínuo linear de bytes. Enviar o array 'achatado' (flatten) como um buffer `sendbuf` garante que a transferência seja um bloco contíguo puro (array plano 1D). O `Gatherv` joga isso direto no vetor de resposta no rank 0 (`saida_flat`) sem precisar fazer ginástica de conversão ou malabarismo de ponteiros em C. Após a coleta total em 1D, apenas o Rank 0 faz um `.reshape(H, W)` que é quase instantâneo para reconstruir a imagem 2D."

### ❓ 5. Professora: "O que é esse IQR (Interquartile Range) e por que vocês usaram isso?"
**R:** "O IQR é uma métrica estatística robusta (melhor que o desvio padrão puro) para identificar 'outliers' - pontos de medição fora da curva. Em computação paralela num sistema operacional, sempre há picos inexplicáveis de demora. Uma thread do SO pode entrar na frente e interromper nosso processo por milissegundos. Se num cenário rodamos 30 vezes e 29 deram 0.1s e 1 deu 2.0s por causa do SO, essa execução de 2.0s iria explodir a média. O IQR (Q3 - Q1) acha a faixa normal, exclui esse ponto absurdo, garantindo que o speedup calculado seja íntegro."

### ❓ 6. Professora: "Se eu tiver um cluster com 10 máquinas, cada máquina com apenas 1 núcleo fraco, como o MPI lida com isso em comparação a ter uma máquina com 10 núcleos?"
**R:** "Do ponto de vista da nossa programação, não muda nada, porque usamos MPI, que foi criado exatamente para suportar a arquitetura de Memória Distribuída. A mesma aplicação que roda dividindo a RAM local em uma máquina multicore, rodará dividindo blocos entre máquinas distintas por rede. O que mudaria na prática seria o **overhead de comunicação**. A placa mãe é muito mais veloz que a placa de rede Gigabit. Num cluster de rede lenta, o envio via `Bcast` da imagem de vários Megabytes demoraria mais, o que puxaria o Speedup total para baixo."

---
<br>

## CONCLUSÃO E DICAS PARA A APRESENTAÇÃO

- Mostrem os gráficos que foram gerados na pasta `output/` (`speedup_vs_processos.png` e `tempo_vs_processos.png`).
- Expliquem com clareza a curva do gráfico de speedup. Mostre a linha do Speedup Ideal (linear) e como o filtro de mediana consegue acompanhá-la melhor do que o de média, embasando com a explicação da proporção de cálculo vs overhead de comunicação.
- Na hora do código, exibam a elegância da regra das **maiúsculas** (`comm.Bcast`, `comm.Gatherv`) e o uso de Buffers NumPy. Isso brilha aos olhos do professor.
- Mostrem as imagens resultantes, provem que o ruído saiu e que não há uma "fita de erro" horizontal na imagem devido ao tratamento perfeito de bordas (halo).

Boa sorte! Com esse material e o código pronto em mãos, a apresentação tem tudo para receber nota máxima.

# Explicação Begezinha do Trabalho de MPI

Se liga, esse documento é tipo um "preparacao.md" mas sem o formalismo. É a visão geral do trampo, explicada de um jeito que qualquer um entende. Bora?

---

## SUMÁRIO

1. [O que é esse rolê?](#1-o-que-é-esse-rolê)
2. [MPI: O que é e como funciona](#2-mpi-o-que-é-e-como-funciona)
3. [Imagem e Filtros: a parte de foto](#3-imagem-e-filtros-a-parte-de-foto)
4. [O Código: o que cada arquivo faz](#4-o-código-o-que-cada-arquivo-faz)
5. [Fluxo do Programa: passo a passo](#5-fluxo-do-programa-passo-a-passo)
6. [As 4 Perguntas Reflexivas](#6-as-4-perguntas-reflexivas)
7. [FAQ: perguntas que a professora pode fazer](#7-faq-perguntas-que-a-professora-pode-fazer)

---

## 1. O que é esse rolê?

É um trabalho de **Programação Paralela** onde a gente pega uma imagem, aplica dois filtros nela (média e mediana), e compara o desempenho entre:

- **Versão sequencial**: um processador fazendo tudo sozinho.
- **Versão paralela**: vários processadores (ou máquinas) dividindo o serviço usando **MPI**.

A gente roda com 1, 2, 4, 8 processos e vê qual ganha mais velocidade (speedup).

---

## 2. MPI: O que é e como funciona

### O que é MPI?

MPI = **Message Passing Interface**. É um **padrão** (especificação) que define como processos trocam mensagens entre si. Não é uma biblioteca em si — as bibliotecas que implementam o padrão são Open MPI, MPICH, Intel MPI, etc.

No Python a gente usa o pacote `mpi4py` pra falar com o MPI.

### Memória Distribuída

Diferente de threads tradicionais (que compartilham memória), no MPI cada processo tem **sua própria memória RAM**. Se o Processo 0 quer dar um dado pro Processo 1, ele precisa **enviar uma mensagem** (igual mandar um zap). Não tem variável compartilhada.

### Conceitos básicos

| Termo | O que é |
|-------|---------|
| `MPI_COMM_WORLD` | O "grupo" de todos os processos rodando no programa |
| **Rank** | O ID de cada processo (0, 1, 2, 3...) |
| **Size** | Quantos processos tem no total |
| **Bcast** | Um processo manda um dado pra TODO MUNDO (broadcast) |
| **Scatter** | Divide um vetor grande em pedaços e distribui entre os processos |
| **Gather** | Junta os pedaços de volta num vetor só |
| **Gatherv** | Gather que aceita pedaços de tamanhos diferentes (útil!) |
| **Barrier** | "Travinha" que espera todo mundo chegar pra continuar |

### Regra de Ouro do mpi4py: Maiúscula vs Minúscula

Isso é **CRUCIAL** pro trabalho:

- **Métodos com letra minúscula** (`comm.bcast`, `comm.send`): servem pra enviar QUALQUER objeto Python (lista, dicionário). Mas eles usam `pickle` pra serializar, o que é **BEEEEM lento** pra dados grandes.
- **Métodos com letra MAIÚSCULA** (`comm.Bcast`, `comm.Gatherv`): funcionam com **buffer numpy** puro. Não tem serialização — manda os bytes brutos direto. **É MUITO MAIS RÁPIDO.**

No trabalho, a gente usa `comm.Bcast` pra mandar a imagem (milhões de bytes) e `comm.Gatherv` pra juntar os resultados. Usar minúscula deixaria o bagaço lento.

### Comunicação Ponto-a-Ponto vs Coletiva

- **Ponto-a-ponto**: um processo envia, outro recebe (`Send`/`Recv`).
- **Coletiva**: todo mundo participa (`Bcast`, `Scatter`, `Gather`, `Reduce`, `Barrier`). No trabalho usamos só operações coletivas pra distribuir e coletar os pedaços da imagem.

---

## 3. Imagem e Filtros: a parte de foto

### Como o computador enxerga uma imagem?

Pra ele, imagem é uma **matriz de números**. Cada número é um pixel de 0 (preto) a 255 (branco). Os valores no meio são tons de cinza.

### Filtro de Média 3x3

Pega uma janela 3x3 em volta de cada pixel, soma os 9 valores e divide por 9. O resultado substitui o pixel central.

**Efeito**: borrão (blur). Tira ruído gaussiano mas perde nitidez.

### Filtro de Mediana 3x3

Pega os 9 valores da janela 3x3, **ordena** do menor pro maior e pega o do meio (5º valor).

**Efeito**: ótimo pra tirar ruído "sal e pimenta" (pixels brancos ou pretos aleatórios). Preserva melhor as bordas.

**Custo**: ordenar 9 números pra cada pixel de uma imagem de 16 milhões de pixels é pesado. Por isso a paralelização compensa mais pra mediana do que pra média.

---

## 4. O Código: o que cada arquivo faz

O projeto tem 3 arquivos:

### `filters.py` — O coração da parada

Tem as funções que aplicam os filtros de verdade:

- `apply_mean_filter_fast` / `apply_median_filter_fast`: versões sequenciais usadas como **baseline** (referência de tempo).
- `compute_mean_block` / `compute_median_block`: versões que processam **blocos** (pedaços) da imagem. São usadas na versão paralela.
  - Elas recebem o bloco **com linhas de halo** (explico já já) e retornam só as linhas que pertencem àquele processo.

### `benchmark.py` — O estatístico

Cuida de:

- **Remover outliers** com IQR (Intervalo Interquartil): joga fora as medições muito estranhas que podem ter sido causadas pelo sistema operacional.
- **Calcular estatísticas**: média, variância, desvio padrão.
- **Calcular speedup**: `TempoSequencial / TempoParalelo`.
- **Gerar gráficos**: `tempo_vs_processos.png` e `speedup_vs_processos.png`.
- **Gerar relatório**: `output/relatorio.txt` com tabela e respostas reflexivas.
- **Salvar CSV**: `output/benchmark_results.csv` pra abrir no Excel.

### `main.py` — O maestro

Orquestra tudo. É o arquivo que você roda com:

```bash
mpiexec -n 4 python main.py
```

(número de processos pode ser 1, 2, 4, 8...)

---

## 5. Fluxo do Programa: passo a passo

Aqui vai o que acontece quando você roda o bagaço:

### 1. Inicialização
O MPI descobre qual é seu rank (0, 1, 2...) e quantos processos tem no total.

### 2. Carregar a imagem
Só o **Rank 0** faz isso. Ele procura `entrada.jpg`. Se não achar, gera uma imagem sintética de **4000x4000 pixels** com bastante ruído (pra garantir que o processamento seja pesado o suficiente pra justificar o paralelismo).

### 3. Distribuir a imagem
- Rank 0 manda a altura e largura pra geral via `comm.bcast` (só 2 números, usar maiúscula aqui não faz diferença).
- Todo mundo aloca um array do tamanho certo.
- Rank 0 manda a imagem COMPLETA pra todo mundo via `comm.Bcast` (MAIÚSCULA, buffer numpy, rapidão).

**Por que mandar a imagem completa pra todo mundo?** Porque aí cada processo pode pegar as linhas de halo (vizinhança) sem precisar pedir pro vizinho. É mais simples.

### 4. Benchmark Sequencial (baseline)
Só Rank 0 roda `apply_mean_filter_fast` e `apply_median_filter_fast` na imagem inteira. Faz **30 repetições** com warm-up antes pra aquecer o cache. Guarda os tempos.

### 5. Calcular as fatias
A função `calcular_fatias` divide as linhas da imagem entre os processos. Se a altura não divide exato, os primeiros processos ganham 1 linha extra. Exemplo: 4000 linhas com 3 processos:
- Processo 0: 1334 linhas
- Processo 1: 1333 linhas
- Processo 2: 1333 linhas

### 6. Benchmark Paralelo (30 repetições pra cada filtro)
Em cada iteração:

1. `comm.Barrier()` — todo mundo espera na linha de largada.
2. Cada processo recorta seu pedaço da imagem (com as linhas de halo).
3. Aplica o filtro no pedaço.
4. `comm.Gatherv` — Rank 0 coleta os resultados de todo mundo num array 1D contínuo.
5. `comm.Barrier()` — espera todo mundo terminar.
6. O tempo da iteração é o **MAIOR** entre todos os processos (porque ninguém termina antes do mais lento).

### 7. Halo: o tratamento de bordas entre processos

**Problema**: se cada processo pega suas linhas e aplica o filtro, a primeira linha de cada bloco não tem vizinho de cima (e a última não tem vizinho de baixo). Daria erro nas fronteiras.

**Solução**: cada processo pega **1 linha extra acima e 1 abaixo** do seu bloco (só pra consulta). Essas são as **linhas de halo**. Depois de aplicar o filtro, ele descarta essas linhas extras e manda só as linhas que realmente são dele.

**Por que funciona de boa?** Porque a imagem inteira já foi distribuída pra todo mundo via `comm.Bcast`. Pegar a linha do vizinho é só um slice no array local, nenhuma comunicação extra necessária.

### 8. Salvar resultados
Rank 0 salva:
- Imagens processadas na pasta `output/`
- Gráficos de tempo e speedup
- Relatório em texto
- CSV
- `historico.json` — salva resultados acumulados de várias execuções

---

## 6. As 4 Perguntas Reflexivas

Essas são as perguntas que a professora pediu pra responder no relatório. Você PRECISA saber explicar cada uma de cabeça na apresentação.

### 1. O filtro de mediana paralelo teve speedup maior ou menor que o de média? Por quê?

**Resposta curta**: A mediana teve speedup **MAIOR**.

**A explicação completa**:

Pensa numa balança: de um lado o **trabalho** que a CPU faz, do outro o **custo de comunicação** do MPI (enviar imagem, coletar resultados, sincronizar).

- **Filtro de média**: é mó tranquilo pro processador - só 9 somas e 1 divisão por pixel. O bagaço é leve. Quando você divide entre vários processos, o tempo que você GANHA dividindo o trabalho é pequeno. Já o tempo que você PERDE com comunicação (Bcast, Gatherv, Barrier) continua ali, fixo. Resultado: o speedup não cresce tanto porque o overhead de comunicação come uma parte grande do ganho.

- **Filtro de mediana**: esse é pesadão. Pra cada pixel, ele precisa ordenar 9 números (e ordenação é cara em termos de processamento). O trabalho total é **~7× maior** que o da média. Aí quando você divide entre 8 processos, cada um faz bem menos trabalho, e o ganho é ENORME. O custo de comunicação é o mesmo, mas ele fica irrelevante perto do trabalho economizado.

**Na prática com nossos números reais**:
- Média: 0.25s sequencial → 0.04s com 8 proc = speedup **6.26×**
- Mediana: 1.72s sequencial → 0.22s com 8 proc = speedup **7.82×** (quase o ideal de 8×!)

**Resumo em uma frase**: quanto mais pesada a tarefa original, mais vantagem você tira da paralelização, porque o custo fixo da comunicação pesa menos no total.

### 2. O que acontece com os pixels nas fronteiras entre processos? Como tratamos?

**O problema**:

Imagina que você tem uma foto e corta ela em 4 tiras horizontais. Cada tira vai pra um processo diferente.

Pra calcular o filtro 3x3 num pixel, você precisa olhar pros vizinhos de cima e de baixo. Agora pensa no primeiro pixel da **tira do Processo 2**: o vizinho de cima dele está na **tira do Processo 1**! Se cada processo só tem o próprio pedaço, ele não consegue calcular a borda superior da tira dele. O resultado final fica todo errado — aparece um "risco" horizontal no meio da imagem.

**A solução: Linhas de Halo (Ghost Cells)**

A gente faz o seguinte: cada processo pega **um pouco mais** do que a fatia dele. Se o Processo 1 é responsável pelas linhas 1000~1999, ele na verdade puxa as linhas **999~2000** (1 linha extra de cada lado). Essas linhas extras são os **halos**.

Com o halo:
- O pixel 1000 tem o vizinho de cima (linha 999) disponível.
- O pixel 1999 tem o vizinho de baixo (linha 2000) disponível.
- O filtro calcula certinho pra todas as linhas da fatia original.
- Depois do cálculo, o processo **descarta** as linhas de halo e manda só as linhas legítimas (1000~1999) pra serem juntadas.

**Por que isso é de graça no nosso código?** Porque no início do programa, a gente mandou a **imagem inteira** pra todo mundo via `comm.Bcast`. Cada processo tem a foto completa na memória dele. Pegar a linha de halo é só fazer `image[halo_top:halo_bot, :]` — um simples slice no numpy, sem precisar enviar mensagem extra pros vizinhos. É uma solução elegante e eficiente.

### 3. Por que `comm.Bcast` (maiúscula) e não `comm.bcast` (minúscula)?

**Resposta**: Porque a maiúscula é **MUITO mais rápida** pra enviar dados grandes como uma imagem.

**A explicação técnica sem complicação**:

O mpi4py tem dois modos de enviar dados:

**Maiúscula (`comm.Bcast`, `comm.Gatherv`, etc.)**:
- Trabalha com **buffer numpy puro** — aquela região de memória onde o array numpy tá guardado.
- O MPI pega os bytes brutos da RAM e manda pela rede. **Não tem conversão, não tem serialização, não tem nada no meio.**
- É tão rápido quanto fazer isso em C puro.
- Perfeito pra imagens, que são arrays gigantes de números.

**Minúscula (`comm.bcast`, `comm.gather`, etc.)**:
- Aceita QUALQUER objeto Python (lista, dicionário, string, o que for).
- Mas pra isso, ele precisa **serializar** o objeto numa sequência de bytes usando `pickle`.
- Aí manda esses bytes serializados pela rede.
- Do outro lado, faz o caminho inverso (desserializar).
- Esse processo de empacotar/desempacotar é **lento pra caramba** pra dados grandes.

**O que mudaria se usássemos minúscula?**

Nossa imagem tem 4000×4000 = **16 milhões de pixels** (~16 MB). Serializar 16 MB com pickle, mandar, e desserializar do outro lado adicionaria um overhead enorme. O tempo de comunicação pularia de alguns milissegundos pra **centenas de milissegundos ou até segundos**. 

Resultado: o tempo paralelo ficaria maior que o sequencial, e o speedup iria pro espaço (tipo 0.5× — ou seja, mais lento que não usar MPI).

**Regra de bolso**: objeto pequeno (2 inteiros) → pode usar minúscula sem medo. Objeto grande (imagem inteira) → **sempre maiúscula**.

### 4. Se o número de processos não divide exatamente a altura da imagem, o que o código faz?

**Resposta**: Distribui as linhas extras entre os primeiros processos, de forma equilibrada.

**Exemplo real**: altura = 4000 linhas, 3 processos.

4000 ÷ 3 = 1333, **resto 1**. Isso quer dizer que não dá pra dividir igual. Se cada processo pegar exatamente 1333 linhas, a soma dá 3999 — vai faltar 1 linha no final.

O código resolve assim:
```python
base = 4000 // 3   # = 1333 (todo mundo pega isso)
resto = 4000 % 3   # = 1 (1 linha sobrando)

# Distribui o resto: os primeiros processos ganham +1 linha
# Rank 0: 1333 + 1 = 1334 linhas
# Rank 1: 1333 + 0 = 1333 linhas
# Rank 2: 1333 + 0 = 1333 linhas
# Total: 1334 + 1333 + 1333 = 4000 ✅
```

**Por que essa abordagem é boa?**
- Todo mundo fica com uma quantidade parecida de trabalho (ninguém faz muito mais que os outros).
- O `Gatherv` consegue lidar com isso porque aceita cada processo mandando um tamanho diferente (diferente do `Gather` normal que exige todo mundo igual).
- A diferença de no máximo 1 linha entre processos é insignificante pra performance.

**E se a imagem for menor que o número de processos?** A função também trata isso — cada processo pega pelo menos 1 linha, e se sobrarem processos sem nada pra fazer, eles simplesmente não recebem linhas. Nosso código com 4000 linhas e até 8 processos nunca vai ter esse problema, mas a lógica funciona pra qualquer combinação.

---

## 7. FAQ: perguntas que a professora pode fazer

### "Por que Gatherv e não Gather?"
Porque as fatias podem ter **tamanhos diferentes** (quando a altura não divide exato). `Gather` exige que todo mundo mande o mesmo número de bytes. `Gatherv` aceita cada um mandando um tamanho diferente.

### "Pra que serve o Barrier no loop?"
Pra garantir que todo mundo comece ao mesmo tempo e que a gente só pare o cronômetro quando **todos** terminaram. Senão um processo pode começar antes e o tempo medido fica errado.

### "Por que 30 repetições dentro do código e não 30 mpiexec?"
A professora pediu explicitamente. Se rodar `mpiexec` 30 vezes, cada execução tem um "início frio" (criar topologia, alocar memória, etc.) que contamina a medição. Rodando 30 iterações dentro de um único `mpiexec`, a gente elimina esse custo fixo e mede só o processamento puro.

### "Por que achatar o resultado com `.flatten()` no Gatherv?"
O MPI em C entende memória como um **bloco contínuo de bytes**. Mandar o array 2D achatado (1D) garante que a transferência seja um bloco contíguo sem gaps. O rank 0 faz `.reshape(H, W)` no final e pronto.

### "O que é IQR e pra que serve?"
IQR (Interquartile Range) é uma forma de identificar **outliers**. Pega os tempos, acha o 25º percentil (Q1) e o 75º (Q3). A diferença entre eles é o IQR. Qualquer tempo que fuja de `Q1 - 1.5*IQR` até `Q3 + 1.5*IQR` é considerado outlier e removido.

Isso evita que uma lentidão aleatória do sistema operacional (antivírus, background update) foda a média.

### "Qual a diferença entre rodar numa máquina com 10 núcleos vs 10 máquinas?"
Pra lógica do programa, **nenhuma**. MPI foi feito pra memória distribuída, então funciona igual. O que muda é a **velocidade da comunicação**: placa-mãe é mais rápida que rede Gigabit. Num cluster de verdade o overhead de comunicação seria maior.

---

## 8. Analisando os Resultados Obtidos

Aqui vou usar os números **reais** que o benchmark gerou pra explicar o que cada um significa. Abre o `output/benchmark_results.csv` ou `output/relatorio.txt` enquanto lê.

### Tabela Resumo dos Resultados Reais

| Filtro | Processos | Tempo Médio (s) | Speedup |
|--------|-----------|-----------------|---------|
| Média  | 1 (seq)   | 0.2476 s        | 1.00×   |
| Média  | 2         | 0.0848 s        | 2.92×   |
| Média  | 4         | 0.0588 s        | 4.21×   |
| Média  | 8         | 0.0396 s        | 6.26×   |
| Mediana| 1 (seq)   | 1.7219 s        | 1.00×   |
| Mediana | 2         | 0.6573 s        | 2.62×   |
| Mediana | 4         | 0.3869 s        | 4.45×   |
| Mediana | 8         | 0.2202 s        | 7.82×   |

### O que esses números estão dizendo?

#### 1. A mediana é MUITO mais pesada que a média

No sequencial (1 processo só):
- **Média**: 0.25 segundos
- **Mediana**: 1.72 segundos → **~7× mais lenta**

Isso faz sentido: a média é 9 somas + 1 divisão por pixel. A mediana precisa **ordenar** 9 números pra cada pixel — isso é bem mais custoso. É exatamente por isso que a paralelização vale mais a pena pra mediana.

#### 2. Speedup: mediana ganha de lavada com mais processos

Olha a evolução:

| Processos | Speedup Média | Speedup Mediana |
|-----------|---------------|-----------------|
| 2         | 2.92×         | 2.62×           |
| 4         | 4.21×         | 4.45×           |
| 8         | 6.26×         | **7.82×**       |

Com **2 processos** a média ainda ganha (2.92 > 2.62), mas a partir de **4 processos** a mediana passa na frente. Com **8 processos** a mediana chega em **7.82×** de speedup — muito perto do ideal (8×), enquanto a média fica em 6.26×.

**Por que isso acontece?** O filtro de média é tão leve que o overhead de comunicação (enviar os dados, sincronizar, coletar resultados) começa a pesar no tempo total. Já a mediana é tão pesada que o ganho de dividir o trabalho entre mais CPUs compensa MUITO mais.

Pensa assim:
- Média: 0.25s de trabalho + 0.05s de comunicação → speedup limitado
- Mediana: 1.72s de trabalho + 0.05s de comunicação → comunicação quase irrelevante

#### 3. Nenhum dos dois atinge speedup perfeito

O speedup **ideal** (ou linear) seria:
- 2 processos → 2× mais rápido
- 4 processos → 4× mais rápido
- 8 processos → 8× mais rápido

Na prática a gente nunca atinge isso por causa de:
- **Overhead de comunicação** (`Bcast`, `Gatherv`, `Barrier` demoram)
- **Parte sequencial** (o Rank 0 faz algumas coisas sozinho, tipo ler a imagem e salvar)
- **Desequilíbrio de carga** (alguns processos podem pegar 1 linha a mais que outros)
- **Lei de Amdahl** — sempre tem uma parte do código que não é paralelizável

#### 4. Variância: o mais fraco usando 2 processos

Repara na variância dos filtros com 2 processos:
- Mediana com 2 processos: **0.0068 s²** (a maior de todas)
- Mediana com 4 processos: 0.0014 s²
- Mediana com 8 processos: 0.00007 s²

Isso mostra que com **2 processos** cada um pega metade da imagem (2000 linhas cada) e o tempo variava bastante entre as execuções. Conforme aumenta o número de processos, cada um faz menos trabalho e os tempos ficam mais consistentes (menor variância).

#### 5. Quantas amostras foram válidas?

O IQR removeu alguns outliers de cada execução:
- Média: 24 a 30 amostras válidas (de 30)
- Mediana: 24 a 30 amostras válidas (de 30)

Nada preocupante — o IQR removeu no máximo 6 execuções em 30, o que é normal.

### Interpretação Visual

Dá uma olhada nos gráficos em `output/`:

**`tempo_vs_processos.png`**: mostra o tempo caindo conforme aumenta o número de processos. A curva da mediana começa lá em cima (~1.72s) e despenca rápido. A média já começa baixa (0.25s) e desce mais devagar.

**`speedup_vs_processos.png`**: mostra o speedup subindo. A linha ideal é uma diagonal perfeita (45°). A mediana acompanha essa diagonal bem de perto. A média vai se afastando conforme adiciona processos.

### Resumo pra Apresentação

Se a professora perguntar "e aí, funcionou?", mostra a tabela e fala:

1. **Funcionou sim — speedup positivo em todos os casos.** Com 8 processos, os dois filtros ficaram mais rápidos.
2. **A mediana se deu melhor** porque o trabalho dela é mais pesado, então dividir compensa mais.
3. **Com 2 processos a média ainda ganha** porque o overhead de comunicação não compensa o ganho pequeno.
4. **O tratamento de bordas funcionou** — as imagens não têm "riscos" no meio (prova: as saídas em `output/`).

## Conclusão e Dicas pra Apresentação

- **Mostre os gráficos** da pasta `output/` — eles são a prova do trabalho.
- **Explique a curva do speedup**: porque a mediana acompanha melhor a linha ideal do que a média.
- **Enfatize o uso de maiúsculas** (`comm.Bcast`, `comm.Gatherv`) — isso mostra que vocês entenderam de verdade.
- **Mostre as imagens** — prova que o negócio funciona e não tem "risco" no meio (halo funcionando).

Boa sorte na apresentação! Se manja.

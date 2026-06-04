import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem janela (compatível com MPI)
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def remover_outliers_iqr(tempos: List[float]) -> List[float]:
    #remove outliers de uma lista de tempos usando a regra do IQR
    vetor_tempos = np.array(tempos, dtype=float)
    q1 = np.percentile(vetor_tempos, 25)
    q3 = np.percentile(vetor_tempos, 75)
    aiq = q3 - q1
    limite_inferior = q1 - 1.5 * aiq
    limite_superior = q3 + 1.5 * aiq
    filtrados = vetor_tempos[(vetor_tempos >= limite_inferior) & (vetor_tempos <= limite_superior)]
    return filtrados.tolist()


def calcular_estatisticas(tempos: List[float]) -> Dict[str, float]:
    #calcula média, variância e desvio padrão de uma lista de tempos
    limpos = remover_outliers_iqr(tempos)
    vetor_limpo = np.array(limpos, dtype=float)
    return {
        "media":         float(np.mean(vetor_limpo)),
        "variancia":     float(np.var(vetor_limpo)),
        "desvio_padrao": float(np.std(vetor_limpo)),
        "n":             len(limpos),
        "n_bruto":       len(tempos),
    }


def calcular_speedup(media_seq: float, media_par: float) -> float:
    if media_par <= 0:
        return 0.0
    return media_seq / media_par


def salvar_csv(resultados: Dict, diretorio_saida: str = "output") -> str:
    #salva a tabela de resultados em csv
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_arquivo = os.path.join(diretorio_saida, "resultados_benchmark.csv")

    with open(caminho_arquivo, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow([
            "Filtro", "Nº Processos", "Tempo Médio (s)",
            "Variância (s²)", "Desvio Padrão (s)", "Speedup", "N amostras"
        ])
        for filtro, dados_proc in resultados.items():
            media_seq = dados_proc.get(1, {}).get("media", None)
            for n_proc in sorted(dados_proc.keys()):
                estatisticas = dados_proc[n_proc]
                speedup = 1.0 if n_proc == 1 else (
                    calcular_speedup(media_seq, estatisticas["media"]) if media_seq else 0.0
                )
                escritor.writerow([
                    filtro, n_proc,
                    f"{estatisticas['media']:.6f}",
                    f"{estatisticas['variancia']:.8f}",
                    f"{estatisticas['desvio_padrao']:.6f}",
                    f"{speedup:.4f}",
                    estatisticas["n"],
                ])

    print(f"[benchmark] CSV salvo em: {caminho_arquivo}")
    return caminho_arquivo


def gerar_grafico_tempo_vs_processos(resultados: Dict, diretorio_saida: str = "output") -> str:
    #gera o gráfico de tempo médio por número de processos para cada filtro
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_arquivo = os.path.join(diretorio_saida, "tempo_vs_processos.png")

    figura, eixos = plt.subplots(figsize=(8, 5))

    cores      = {"media": "#2196F3", "mediana": "#FF5722"}
    marcadores = {"media": "o",       "mediana": "s"}
    rotulos    = {"media": "Filtro de Média",  "mediana": "Filtro de Mediana"}

    for filtro, dados_proc in resultados.items():
        processos = sorted(dados_proc.keys())
        tempos    = [dados_proc[p]["media"] for p in processos]
        desvios   = [dados_proc[p]["desvio_padrao"] for p in processos]
        
        eixos.errorbar(
            processos, tempos,
            yerr=desvios,
            label=rotulos.get(filtro, filtro),
            color=cores.get(filtro, "gray"),
            marker=marcadores.get(filtro, "o"),
            linewidth=2,
            markersize=8,
            capsize=5,
        )

    eixos.set_xlabel("Número de Processos MPI", fontsize=12)
    eixos.set_ylabel("Tempo Médio (s)", fontsize=12)
    eixos.set_title("Tempo de Execução vs. Número de Processos", fontsize=14, fontweight="bold")
    eixos.legend(fontsize=11)
    eixos.grid(True, linestyle="--", alpha=0.5)
    eixos.set_xticks(sorted(set(p for fd in resultados.values() for p in fd.keys())))

    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)
    print(f"[benchmark] Gráfico salvo em: {caminho_arquivo}")
    return caminho_arquivo


def gerar_grafico_speedup(resultados: Dict, diretorio_saida: str = "output") -> str:
    #Gera gráfico de speedup por número de processos para cada filtro
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_arquivo = os.path.join(diretorio_saida, "speedup_vs_processos.png")

    figura, eixos = plt.subplots(figsize=(8, 5))

    cores      = {"media": "#2196F3", "mediana": "#FF5722"}
    marcadores = {"media": "o",       "mediana": "s"}
    rotulos    = {"media": "Filtro de Média",  "mediana": "Filtro de Mediana"}

    todos_processos = set()
    for filtro, dados_proc in resultados.items():
        media_seq = dados_proc.get(1, {}).get("media", None)
        if media_seq is None:
            continue
        processos = sorted(dados_proc.keys())
        lista_speedups = []
        for p in processos:
            s = calcular_speedup(media_seq, dados_proc[p]["media"]) if p != 1 else 1.0
            lista_speedups.append(s)
        todos_processos.update(processos)
        eixos.plot(
            processos, lista_speedups,
            label=rotulos.get(filtro, filtro),
            color=cores.get(filtro, "gray"),
            marker=marcadores.get(filtro, "o"),
            linewidth=2,
            markersize=8,
        )

    # Linha de speedup ideal
    max_proc = max(todos_processos) if todos_processos else 8
    processos_ideais = list(range(1, max_proc + 1))
    eixos.plot(
        processos_ideais, processos_ideais,
        label="Speedup Ideal",
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )

    eixos.set_xlabel("Número de Processos MPI", fontsize=12)
    eixos.set_ylabel("Speedup", fontsize=12)
    eixos.set_title("Speedup vs. Número de Processos", fontsize=14, fontweight="bold")
    eixos.legend(fontsize=11)
    eixos.grid(True, linestyle="--", alpha=0.5)
    eixos.set_xticks(sorted(todos_processos))

    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)
    print(f"[benchmark] Gráfico de speedup salvo em: {caminho_arquivo}")
    return caminho_arquivo



def gerar_relatorio(resultados: Dict, dimensoes_imagem: Tuple[int, int],
                    origem_imagem: str, diretorio_saida: str = "output") -> str:

    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_arquivo = os.path.join(diretorio_saida, "relatorio.txt")

    altura, largura = dimensoes_imagem

    # Monta a tabela de resultados
    linhas_tabela = []
    linhas_tabela.append(
        f"{'Filtro':<10} {'Processos':>10} {'Tempo Médio (s)':>18} "
        f"{'Variância (s²)':>16} {'Desvio Padrão (s)':>18} {'Speedup':>10}"
    )
    linhas_tabela.append("-" * 90)
    
    for filtro, dados_proc in resultados.items():
        media_seq = dados_proc.get(1, {}).get("media", None)
        nome = "Média" if filtro == "media" else "Mediana"
        for n_proc in sorted(dados_proc.keys()):
            estatisticas = dados_proc[n_proc]
            speedup = 1.0 if n_proc == 1 else (
                calcular_speedup(media_seq, estatisticas["media"]) if media_seq else 0.0
            )
            rotulo = "Sequencial" if n_proc == 1 else f"{n_proc} processos"
            linhas_tabela.append(
                f"{nome:<10} {rotulo:>10} {estatisticas['media']:>18.6f} "
                f"{estatisticas['variancia']:>16.8f} {estatisticas['desvio_padrao']:>18.6f} {speedup:>10.4f}"
            )
        linhas_tabela.append("")

    tabela_str = "\n".join(linhas_tabela)

    # Respostas reflexivas dinâmicas baseadas nos dados reais
    speedups_media   = {}
    speedups_mediana = {}
    for n_proc in [2, 4, 8]:
        if "media" in resultados and n_proc in resultados["media"] and 1 in resultados["media"]:
            speedups_media[n_proc] = calcular_speedup(
                resultados["media"][1]["media"], resultados["media"][n_proc]["media"]
            )
        if "mediana" in resultados and n_proc in resultados["mediana"] and 1 in resultados["mediana"]:
            speedups_mediana[n_proc] = calcular_speedup(
                resultados["mediana"][1]["media"], resultados["mediana"][n_proc]["media"]
            )

    # Comparação de speedup médio entre os filtros
    texto_comparacao_speedup = ""
    if speedups_media and speedups_mediana:
        media_dos_speedups_media   = np.mean(list(speedups_media.values()))
        media_dos_speedups_mediana = np.mean(list(speedups_mediana.values()))
        if media_dos_speedups_mediana > media_dos_speedups_media:
            texto_comparacao_speedup = (
                f"O filtro de mediana obteve speedup médio ({media_dos_speedups_mediana:.3f}) MAIOR "
                f"que o de média ({media_dos_speedups_media:.3f}). Isso pode parecer contraintuitivo, "
                f"pois a mediana envolve ordenação e é computacionalmente mais pesada, "
                f"mas significa que o ganho relativo com paralelização foi maior – "
                f"possivelmente porque o overhead do trabalho extra se beneficia mais "
                f"da distribuição entre processos."
            )
        else:
            texto_comparacao_speedup = (
                f"O filtro de média obteve speedup médio ({media_dos_speedups_media:.3f}) MAIOR "
                f"que o de mediana ({media_dos_speedups_mediana:.3f}). Isso é esperado: o filtro "
                f"de mediana requer ordenação dos 9 vizinhos para cada pixel, "
                f"operação que envolve mais sincronização de dados e overhead de "
                f"comunicação relativo, reduzindo o ganho com paralelização."
            )

    relatorio = f"""
================================================================================
         RELATÓRIO – COMPARAÇÃO DE FILTROS DE SUAVIZAÇÃO DE IMAGEM COM MPI
================================================================================

Gerado automaticamente pelo programa em {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
1. INTRODUÇÃO
================================================================================

Este relatório apresenta os resultados da atividade avaliativa de Programação
Paralela com MPI (Message Passing Interface), cujo objetivo é comparar o
desempenho dos filtros de suavização de imagem – filtro de média 3x3 e filtro
de mediana 3x3 – nas versões sequencial e paralela com MPI.

A paralelização é realizada com a biblioteca mpi4py em Python, distribuindo a
imagem entre os processos MPI por faixas horizontais de linhas. As comunicações
coletivas usam buffers numpy (comm.Bcast, comm.Scatterv, comm.Gatherv) para
garantir eficiência em mensagens de grande volume.

================================================================================
2. DESCRIÇÃO DOS FILTROS
================================================================================

2.1 Filtro de Média 3x3
-----------------------
Para cada pixel interno (i, j) da imagem, calcula a média aritmética dos 9
pixels na janela 3x3 centrada em (i, j):

    saída[i,j] = (1/9) * Σ imagem[i+di, j+dj]  para di,dj ∈ {{-1,0,1}}

O resultado é arredondado para o inteiro mais próximo e convertido para uint8.
O filtro de média é simples, eficiente e ideal para suavizar ruído gaussiano,
mas pode borrar bordas/arestas da imagem.

2.2 Filtro de Mediana 3x3
--------------------------
Para cada pixel interno (i, j), extrai os 9 valores da janela 3x3, ordena-os
e seleciona o valor do meio (posição 4, índice 0-based):

    saída[i,j] = median({{imagem[i+di, j+dj] : di,dj ∈ {{-1,0,1}}}})

O filtro de mediana é superior para remoção de ruído impulsivo ("sal e pimenta")
e preserva melhor as bordas, porém é computacionalmente mais custoso que a
média devido à etapa de ordenação.

================================================================================
3. METODOLOGIA
================================================================================

3.1 Imagem Utilizada
---------------------
Origem : {origem_imagem}
Dimensões: {altura} x {largura} pixels (altura x largura), escala de cinza.

Uma imagem grande foi usada para justificar a paralelização – imagens pequenas
têm overhead de comunicação MPI maior do que o ganho de processamento.

3.2 Versão Sequencial
----------------------
A versão sequencial é executada apenas no processo de rank 0. Usa as funções
apply_mean_filter_fast() e apply_median_filter_fast() de filters.py, que
implementam os filtros manualmente com slicing numpy (sem cv2.blur/medianBlur).

3.3 Versão Paralela MPI
------------------------
Etapas:
  1. Rank 0 lê ou gera a imagem em escala de cinza.
  2. Rank 0 transmite o shape da imagem via comm.bcast (apenas 2 inteiros –
     aceitável; a imagem em si usa comm.Bcast com buffer numpy).
  3. A imagem completa é enviada a todos via comm.Bcast(buf, root=0), onde
     buf é um array numpy contíguo – operação eficiente para grandes dados.
  4. Cada processo calcula sua fatia (faixa de linhas) mais linhas de halo.
  5. O processamento é local em cada processo usando compute_mean_block() /
     compute_median_block() de filters.py.
  6. Os resultados são coletados no rank 0 com comm.Gatherv.

3.4 Uso de Bcast/Scatterv/Gatherv com Buffer Numpy
----------------------------------------------------
As operações coletivas com letra maiúscula (comm.Bcast, comm.Scatterv,
comm.Gatherv) operam diretamente sobre buffers numpy sem serialização Python
(pickle), tornando-as ordens de magnitude mais rápidas para arrays grandes.
Usar comm.bcast (minúsculo) para a imagem serializaria o array inteiro com
pickle a cada chamada, adicionando overhead proporcional ao tamanho da imagem.

3.5 Warm-up
-----------
Antes de iniciar as medições, uma execução de aquecimento (warm-up) é
realizada para cada filtro. Isso garante que caches de CPU, alocações de
memória e iniciação de bibliotecas não contaminem as medições.

3.6 30 Execuções por Filtro
----------------------------
Cada filtro é executado 30 vezes em loop interno ao próprio programa MPI,
sem reiniciar o ambiente MPI entre execuções. Os tempos são coletados por
todos os processos e o tempo final de cada iteração é o máximo entre os
processos (o processo mais lento limita o tempo total).

3.7 Uso de comm.Barrier
------------------------
Antes de iniciar o cronômetro de cada iteração, comm.Barrier() sincroniza
todos os processos. Após o processamento, outro comm.Barrier() garante que
todos terminaram antes de registrar o tempo. Isso assegura que o tempo medido
reflita o tempo real de execução paralela.

3.8 Remoção de Outliers por AIQ
---------------------------------
Após coletar os 30 tempos, outliers são removidos pelo método AIQ:
  Q1 = percentil 25, Q3 = percentil 75, AIQ = Q3 – Q1
  Mantém: [Q1 – 1.5·AIQ, Q3 + 1.5·AIQ]
Os tempos restantes são usados para calcular média, variância e desvio padrão.

================================================================================
4. RESULTADOS – TABELA DE TEMPOS E SPEEDUP
================================================================================

{tabela_str}

================================================================================
5. GRÁFICOS
================================================================================

Os gráficos foram gerados na pasta output/:

  • tempo_vs_processos.png  – Tempo médio (±desvio padrão) por nº de processos
  • speedup_vs_processos.png – Speedup observado vs. speedup ideal (linear)

================================================================================
6. RESPOSTAS REFLEXIVAS
================================================================================

6.1 O filtro de mediana paralelo teve speedup maior ou menor que o de média?
------------------------------------------------------------------------------
{texto_comparacao_speedup}

Em termos teóricos: o filtro de mediana tem custo computacional maior por pixel
(requer ordenação de 9 elementos) e, portanto, é mais beneficiado pela
paralelização quando o número de processos é suficiente para compensar o
overhead de comunicação MPI.

6.2 O que acontece com os pixels nas fronteiras entre processos?
-----------------------------------------------------------------
Quando a imagem é dividida em faixas horizontais, o pixel na primeira linha de
uma fatia precisa do valor da última linha da fatia anterior para calcular o
filtro 3x3. Sem tratamento especial, o resultado seria incorreto nas fronteiras.

O código trata isso com "linhas de halo": cada processo recebe sua faixa de
linhas mais 1 linha extra acima (do processo anterior) e/ou 1 linha extra abaixo
(do processo seguinte). Após o processamento, as linhas de halo são descartadas
e apenas os resultados da faixa original são enviados de volta ao rank 0.

Como a imagem completa já está disponível em todos os processos (via Bcast),
o halo é extraído diretamente do array local sem comunicação ponto-a-ponto
adicional.

As bordas externas da imagem (primeira/última linha e coluna) não são
processadas pelo filtro: os valores originais são copiados para a saída,
evitando acessos fora dos limites do array.

6.3 Por que foi usado comm.Bcast e não comm.bcast para enviar a imagem?
------------------------------------------------------------------------
Em mpi4py, existem dois estilos de comunicação coletiva:

  • Minúsculo (comm.bcast): usa pickle para serializar qualquer objeto Python.
    Para um array numpy de {altura}×{largura} pixels ({altura*largura} bytes ≈ {altura*largura/1e6:.1f} MB),
    a serialização e desserialização adicionam overhead significativo (pode ser
    2–5× mais lento que a versão com buffer).

  • Maiúsculo (comm.Bcast): opera diretamente sobre o buffer de memória do array
    numpy, sem serialização. A mensagem MPI contém apenas os bytes brutos do
    array, com desempenho próximo ao limite da banda de memória/rede.

Usar comm.bcast para uma imagem grande tornaria a distribuição um gargalo,
desperdiçando o ganho obtido com a paralelização do processamento.

6.4 Se o número de processos não divide exatamente a altura da imagem,
    o que o código faz?
------------------------------------------------------------------------
Quando altura % n_processos ≠ 0, o código distribui o resto entre os primeiros
processos usando a seguinte lógica:

    base   = altura // n_processos      # linhas mínimas por processo
    resto  = altura %  n_processos      # linhas extras a distribuir

    Processo i recebe: base + (1 se i < resto, senão 0) linhas.

Exemplo: imagem de {altura} linhas com 3 processos:
    base = {altura}//3 = {altura//3}, resto = {altura}%3 = {altura%3}
    Processo 0: {altura//3 + (1 if 0 < altura%3 else 0)} linhas
    Processo 1: {altura//3 + (1 if 1 < altura%3 else 0)} linhas
    Processo 2: {altura//3 + (1 if 2 < altura%3 else 0)} linhas

Isso garante que todos os processos recebam uma quantidade válida de linhas
e que a soma das fatias seja exatamente igual à altura total da imagem.

================================================================================
7. CONCLUSÃO
================================================================================

A paralelização dos filtros de suavização com MPI demonstrou que:

1. O filtro de média, sendo computacionalmente simples (9 somas + divisão),
   tende a ter overhead de comunicação mais significativo em relação ao seu
   trabalho por pixel, limitando o speedup em processadores compartilhados.

2. O filtro de mediana, por ser mais custoso computacionalmente (ordenação),
   apresenta melhor relação trabalho/comunicação e pode atingir speedups mais
   expressivos.

3. O tratamento correto das fronteiras (halos) é essencial para a correção
   dos resultados paralelos – sem ele, pixels nas bordas das fatias seriam
   calculados incorretamente.

4. O uso de operações MPI com buffer numpy (maiúsculas) é fundamental para
   boa performance na transferência de dados de imagem.

5. A metodologia de benchmark (warm-up + 30 repetições + Barrier + IQR)
   proporciona medições estatisticamente confiáveis e comparáveis.

================================================================================
FIM DO RELATÓRIO
================================================================================
""".strip()

    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(relatorio)

    print(f"[benchmark] Relatório salvo em: {caminho_arquivo}")
    return caminho_arquivo
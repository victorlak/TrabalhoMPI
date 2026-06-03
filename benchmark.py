"""
benchmark.py
============
Funções utilitárias para o benchmark dos filtros de suavização.

Responsabilidades:
    - Remoção de outliers pelo método IQR (Interquartile Range).
    - Cálculo de estatísticas: média, variância, desvio padrão.
    - Cálculo de speedup relativo ao tempo sequencial.
    - Salvamento de resultados em CSV e geração de gráficos.
    - Escrita do relatório textual automático.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem janela (compatível com MPI)
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Remoção de outliers por IQR
# ---------------------------------------------------------------------------

def remove_outliers_iqr(times: List[float]) -> List[float]:
    """
    Remove outliers de uma lista de tempos usando a regra do IQR.

    Regra:
        Q1, Q3 = percentis 25 e 75
        IQR = Q3 - Q1
        Mantém valores no intervalo [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

    Parâmetros
    ----------
    times : list[float]
        Lista de tempos de execução em segundos.

    Retorna
    -------
    list[float]
        Lista filtrada sem outliers.
    """
    arr = np.array(times, dtype=float)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = arr[(arr >= lower) & (arr <= upper)]
    return filtered.tolist()


# ---------------------------------------------------------------------------
# Cálculo de estatísticas
# ---------------------------------------------------------------------------

def compute_stats(times: List[float]) -> Dict[str, float]:
    """
    Calcula média, variância e desvio padrão de uma lista de tempos.

    Parâmetros
    ----------
    times : list[float]

    Retorna
    -------
    dict com chaves: 'mean', 'variance', 'std', 'n' (n após remoção de outliers)
    """
    clean = remove_outliers_iqr(times)
    arr = np.array(clean, dtype=float)
    return {
        "mean":     float(np.mean(arr)),
        "variance": float(np.var(arr)),
        "std":      float(np.std(arr)),
        "n":        len(clean),
        "raw_n":    len(times),
    }


def compute_speedup(seq_mean: float, par_mean: float) -> float:
    """
    Calcula o speedup: S = T_seq / T_par.

    Retorna 0.0 se par_mean for zero (evita divisão por zero).
    """
    if par_mean <= 0:
        return 0.0
    return seq_mean / par_mean


# ---------------------------------------------------------------------------
# Salvamento de resultados em CSV
# ---------------------------------------------------------------------------

def save_csv(results: Dict, output_dir: str = "output") -> str:
    """
    Salva a tabela de resultados de benchmark em um arquivo CSV.

    Parâmetros
    ----------
    results : dict
        Estrutura: { filtro: { n_proc: stats_dict } }
        Exemplo: { 'mean': { 1: {'mean':0.5,...}, 2: {...} }, 'median': {...} }
    output_dir : str

    Retorna
    -------
    str : caminho do arquivo CSV criado.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_results.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filtro", "Nº Processos", "Tempo Médio (s)",
            "Variância (s²)", "Desvio Padrão (s)", "Speedup", "N amostras"
        ])
        for filtro, proc_data in results.items():
            seq_mean = proc_data.get(1, {}).get("mean", None)
            for n_proc in sorted(proc_data.keys()):
                stats = proc_data[n_proc]
                speedup = 1.0 if n_proc == 1 else (
                    compute_speedup(seq_mean, stats["mean"]) if seq_mean else 0.0
                )
                writer.writerow([
                    filtro, n_proc,
                    f"{stats['mean']:.6f}",
                    f"{stats['variance']:.8f}",
                    f"{stats['std']:.6f}",
                    f"{speedup:.4f}",
                    stats["n"],
                ])

    print(f"[benchmark] CSV salvo em: {path}")
    return path


# ---------------------------------------------------------------------------
# Geração de gráficos
# ---------------------------------------------------------------------------

def plot_time_vs_procs(results: Dict, output_dir: str = "output") -> str:
    """
    Gera gráfico de tempo médio por número de processos para cada filtro.

    Parâmetros
    ----------
    results : dict (mesma estrutura de save_csv)
    output_dir : str

    Retorna
    -------
    str : caminho da imagem do gráfico.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tempo_vs_processos.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors   = {"mean": "#2196F3", "median": "#FF5722"}
    markers  = {"mean": "o",       "median": "s"}
    labels   = {"mean": "Filtro de Média",  "median": "Filtro de Mediana"}

    for filtro, proc_data in results.items():
        procs = sorted(proc_data.keys())
        times = [proc_data[p]["mean"] for p in procs]
        stds  = [proc_data[p]["std"]  for p in procs]
        ax.errorbar(
            procs, times,
            yerr=stds,
            label=labels.get(filtro, filtro),
            color=colors.get(filtro, "gray"),
            marker=markers.get(filtro, "o"),
            linewidth=2,
            markersize=8,
            capsize=5,
        )

    ax.set_xlabel("Número de Processos MPI", fontsize=12)
    ax.set_ylabel("Tempo Médio (s)", fontsize=12)
    ax.set_title("Tempo de Execução vs. Número de Processos", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(sorted(set(p for fd in results.values() for p in fd.keys())))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[benchmark] Gráfico salvo em: {path}")
    return path


def plot_speedup(results: Dict, output_dir: str = "output") -> str:
    """
    Gera gráfico de speedup por número de processos para cada filtro.
    Inclui a linha de speedup ideal (linear).
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "speedup_vs_processos.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors  = {"mean": "#2196F3", "median": "#FF5722"}
    markers = {"mean": "o",       "median": "s"}
    labels  = {"mean": "Filtro de Média",  "median": "Filtro de Mediana"}

    all_procs = set()
    for filtro, proc_data in results.items():
        seq_mean = proc_data.get(1, {}).get("mean", None)
        if seq_mean is None:
            continue
        procs   = sorted(proc_data.keys())
        speedups = []
        for p in procs:
            s = compute_speedup(seq_mean, proc_data[p]["mean"]) if p != 1 else 1.0
            speedups.append(s)
        all_procs.update(procs)
        ax.plot(
            procs, speedups,
            label=labels.get(filtro, filtro),
            color=colors.get(filtro, "gray"),
            marker=markers.get(filtro, "o"),
            linewidth=2,
            markersize=8,
        )

    # Linha de speedup ideal
    max_p = max(all_procs) if all_procs else 8
    ideal_procs = list(range(1, max_p + 1))
    ax.plot(
        ideal_procs, ideal_procs,
        label="Speedup Ideal",
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("Número de Processos MPI", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title("Speedup vs. Número de Processos", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(sorted(all_procs))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[benchmark] Gráfico de speedup salvo em: {path}")
    return path


# ---------------------------------------------------------------------------
# Geração do relatório textual
# ---------------------------------------------------------------------------

def gerar_relatorio(results: Dict, img_shape: Tuple[int, int],
                    img_source: str, output_dir: str = "output") -> str:
    """
    Gera o arquivo relatorio.txt com os resultados e respostas reflexivas.

    Parâmetros
    ----------
    results : dict
    img_shape : (altura, largura) da imagem usada.
    img_source : 'entrada.jpg (lida do disco)' ou 'imagem sintética gerada'
    output_dir : str

    Retorna
    -------
    str : caminho do relatório gerado.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "relatorio.txt")

    H, W = img_shape

    # Monta a tabela de resultados
    linhas_tabela = []
    linhas_tabela.append(
        f"{'Filtro':<10} {'Processos':>10} {'Tempo Médio (s)':>18} "
        f"{'Variância (s²)':>16} {'Desvio Padrão (s)':>18} {'Speedup':>10}"
    )
    linhas_tabela.append("-" * 90)
    for filtro, proc_data in results.items():
        seq_mean = proc_data.get(1, {}).get("mean", None)
        nome = "Média" if filtro == "mean" else "Mediana"
        for n_proc in sorted(proc_data.keys()):
            stats = proc_data[n_proc]
            speedup = 1.0 if n_proc == 1 else (
                compute_speedup(seq_mean, stats["mean"]) if seq_mean else 0.0
            )
            label = "Sequencial" if n_proc == 1 else f"{n_proc} processos"
            linhas_tabela.append(
                f"{nome:<10} {label:>10} {stats['mean']:>18.6f} "
                f"{stats['variance']:>16.8f} {stats['std']:>18.6f} {speedup:>10.4f}"
            )
        linhas_tabela.append("")

    tabela_str = "\n".join(linhas_tabela)

    # Respostas reflexivas dinâmicas baseadas nos dados reais
    media_speedups   = {}
    mediana_speedups = {}
    for n_proc in [2, 4, 8]:
        if "mean" in results and n_proc in results["mean"] and 1 in results["mean"]:
            media_speedups[n_proc] = compute_speedup(
                results["mean"][1]["mean"], results["mean"][n_proc]["mean"]
            )
        if "median" in results and n_proc in results["median"] and 1 in results["median"]:
            mediana_speedups[n_proc] = compute_speedup(
                results["median"][1]["mean"], results["median"][n_proc]["mean"]
            )

    # Comparação de speedup médio entre os filtros
    comp_speedup_txt = ""
    if media_speedups and mediana_speedups:
        avg_media   = np.mean(list(media_speedups.values()))
        avg_mediana = np.mean(list(mediana_speedups.values()))
        if avg_mediana > avg_media:
            comp_speedup_txt = (
                f"O filtro de mediana obteve speedup médio ({avg_mediana:.3f}) MAIOR "
                f"que o de média ({avg_media:.3f}). Isso pode parecer contraintuitivo, "
                f"pois a mediana envolve ordenação e é computacionalmente mais pesada, "
                f"mas significa que o ganho relativo com paralelização foi maior – "
                f"possivelmente porque o overhead do trabalho extra se beneficia mais "
                f"da distribuição entre processos."
            )
        else:
            comp_speedup_txt = (
                f"O filtro de média obteve speedup médio ({avg_media:.3f}) MAIOR "
                f"que o de mediana ({avg_mediana:.3f}). Isso é esperado: o filtro "
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
Origem : {img_source}
Dimensões: {H} x {W} pixels (altura x largura), escala de cinza.

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

3.8 Remoção de Outliers por IQR
---------------------------------
Após coletar os 30 tempos, outliers são removidos pelo método IQR:
  Q1 = percentil 25, Q3 = percentil 75, IQR = Q3 – Q1
  Mantém: [Q1 – 1.5·IQR, Q3 + 1.5·IQR]
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
{comp_speedup_txt}

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
    Para um array numpy de {H}×{W} pixels ({H*W} bytes ≈ {H*W/1e6:.1f} MB),
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

Exemplo: imagem de {H} linhas com 3 processos:
    base = {H}//3 = {H//3}, resto = {H}%3 = {H%3}
    Processo 0: {H//3 + (1 if 0 < H%3 else 0)} linhas
    Processo 1: {H//3 + (1 if 1 < H%3 else 0)} linhas
    Processo 2: {H//3 + (1 if 2 < H%3 else 0)} linhas

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

    with open(path, "w", encoding="utf-8") as f:
        f.write(relatorio)

    print(f"[benchmark] Relatório salvo em: {path}")
    return path

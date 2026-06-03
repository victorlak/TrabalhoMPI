"""
main.py
=======
Script principal MPI para comparação de filtros de suavização de imagem.

Uso:
    mpiexec -n 1 python main.py   # sequencial (baseline)
    mpiexec -n 2 python main.py
    mpiexec -n 4 python main.py
    mpiexec -n 8 python main.py

O programa detecta automaticamente o número de processos e executa o benchmark
interno (30 repetições + warm-up) sem reiniciar o MPI entre as medições.

Fluxo geral:
  1. Rank 0 lê "entrada.jpg" ou gera imagem sintética 4000×4000 com ruído.
  2. Benchmark sequencial (apenas rank 0, n_proc == 1 ou como baseline).
  3. Benchmark paralelo com MPI:
       a. Rank 0 transmite shape via bcast (apenas 2 inteiros – OK).
       b. Imagem inteira enviada via comm.Bcast (buffer numpy – eficiente).
       c. Cada processo calcula sua fatia (com linhas de halo).
       d. Resultados coletados com comm.Gatherv.
  4. Rank 0 salva imagens, CSV, gráficos e relatório.
"""

import os
import time
import numpy as np
import cv2

from mpi4py import MPI

from filters import (
    apply_mean_filter_fast,
    apply_median_filter_fast,
    compute_mean_block,
    compute_median_block,
)
from benchmark import (
    compute_stats,
    compute_speedup,
    save_csv,
    plot_time_vs_procs,
    plot_speedup,
    gerar_relatorio,
)

# ---------------------------------------------------------------------------
# Configurações globais do benchmark
# ---------------------------------------------------------------------------
N_REPETICOES = 30       # repetições por filtro por configuração
OUTPUT_DIR   = "output" # pasta de saída
IMG_PATH     = "entrada.jpg"
IMG_SINTETICA_SIZE = (4000, 4000)  # altura x largura da imagem sintética

# ---------------------------------------------------------------------------
# Comunicador MPI
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ===========================================================================
# Utilitários
# ===========================================================================

def log(msg: str):
    """Imprime mensagem prefixada com o rank (apenas rank 0 ou com prefixo)."""
    print(f"[rank {rank}] {msg}", flush=True)


def carregar_ou_gerar_imagem() -> tuple[np.ndarray, str]:
    """
    Rank 0: carrega 'entrada.jpg' em escala de cinza ou gera imagem sintética.

    Retorna
    -------
    (image_array, source_description)
    """
    if os.path.exists(IMG_PATH):
        img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log(f"AVISO: não foi possível ler '{IMG_PATH}'. Gerando imagem sintética.")
        else:
            log(f"Imagem carregada: '{IMG_PATH}' ({img.shape[0]}x{img.shape[1]})")
            return img, f"arquivo '{IMG_PATH}' lido do disco"
    # Gera imagem sintética com ruído
    H, W = IMG_SINTETICA_SIZE
    log(f"Gerando imagem sintética {H}x{W} com ruído...")
    rng = np.random.default_rng(seed=42)
    # Base: gradiente suave
    base = np.tile(np.linspace(0, 255, W, dtype=np.float32), (H, 1))
    # Ruído gaussiano
    noise = rng.normal(0, 40, (H, W)).astype(np.float32)
    # Ruído impulsivo ("sal e pimenta")  – beneficia o filtro de mediana
    salt_mask   = rng.random((H, W)) < 0.02
    pepper_mask = rng.random((H, W)) < 0.02
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    img[salt_mask]   = 255
    img[pepper_mask] = 0
    # Salva a imagem gerada para referência
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "entrada_sintetica.png"), img)
    log(f"Imagem sintética salva em '{OUTPUT_DIR}/entrada_sintetica.png'")
    return img, f"imagem sintética gerada ({H}x{W} pixels com ruído gaussiano e impulsivo)"


# ===========================================================================
# Divisão da imagem em fatias para MPI
# ===========================================================================

def calcular_fatias(height: int, n_proc: int) -> list[tuple[int, int]]:
    """
    Calcula os intervalos [start, end) de linhas para cada processo.

    Distribui o resto (height % n_proc) entre os primeiros processos,
    garantindo que todos recebam pelo menos uma linha.

    Parâmetros
    ----------
    height  : altura total da imagem
    n_proc  : número de processos MPI

    Retorna
    -------
    list de (start_row, end_row) para cada rank i in [0, n_proc).
    """
    base  = height // n_proc
    resto = height %  n_proc
    fatias = []
    start = 0
    for i in range(n_proc):
        # Processos 0..resto-1 recebem uma linha extra
        linhas = base + (1 if i < resto else 0)
        fatias.append((start, start + linhas))
        start += linhas
    return fatias


# ===========================================================================
# Versão sequencial do benchmark (executada no rank 0 como baseline)
# ===========================================================================

def benchmark_sequencial(image: np.ndarray) -> dict:
    """
    Executa o benchmark sequencial dos dois filtros no rank 0.

    Retorna dict: { 'mean': [tempos], 'median': [tempos] }
    """
    resultados = {"mean": [], "median": []}
    filter_funcs = {
        "mean":   apply_mean_filter_fast,
        "median": apply_median_filter_fast,
    }

    for nome, func in filter_funcs.items():
        log(f"[SEQ] Warm-up – filtro {nome}...")
        _ = func(image)  # warm-up

        log(f"[SEQ] Benchmark – filtro {nome} ({N_REPETICOES} repetições)...")
        for it in range(N_REPETICOES):
            t0 = time.perf_counter()
            resultado = func(image)
            t1 = time.perf_counter()
            resultados[nome].append(t1 - t0)
            if it == N_REPETICOES - 1:
                # Salva a última imagem processada
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                fname = os.path.join(OUTPUT_DIR, f"seq_{nome}.png")
                cv2.imwrite(fname, resultado)
                log(f"[SEQ] Imagem salva: {fname}")

    return resultados


# ===========================================================================
# Versão paralela – um filtro por vez
# ===========================================================================

def _aplicar_filtro_bloco(bloco_halo, has_top, has_bot, nome_filtro):
    """Despacha para a função de filtro correta."""
    if nome_filtro == "mean":
        return compute_mean_block(bloco_halo, has_top, has_bot)
    else:
        return compute_median_block(bloco_halo, has_top, has_bot)


def benchmark_paralelo_filtro(image_full: np.ndarray, nome_filtro: str) -> list[float]:
    """
    Executa o benchmark paralelo de um filtro específico.

    Passos:
      1. Rank 0 calcula as fatias e transmite via comm.Bcast (buffer numpy).
      2. Cada processo extrai seu bloco + halo do array local.
      3. Aplica o filtro no bloco com halo.
      4. Rank 0 coleta os resultados via Gatherv.
      5. O tempo de cada iteração é o máximo entre todos os processos.

    Parâmetros
    ----------
    image_full : array completo (disponível em todos os processos após Bcast)
    nome_filtro: 'mean' ou 'median'

    Retorna (apenas rank 0)
    -------
    list[float] : tempos de cada iteração (N_REPETICOES)
    """
    H, W = image_full.shape
    fatias  = calcular_fatias(H, size)  # lista de (start, end) por processo
    start_r, end_r = fatias[rank]       # fatia deste processo
    n_linhas = end_r - start_r          # número de linhas sem halo

    # Linhas de halo: 1 linha acima e 1 abaixo (se existirem)
    halo_top = max(0, start_r - 1)
    halo_bot = min(H, end_r + 1)
    has_top_halo = (halo_top < start_r)
    has_bot_halo = (halo_bot > end_r)

    # Prepara estruturas para Gatherv
    # Cada processo contribui com n_linhas * W pixels (uint8)
    send_counts = np.array([
        (fatias[i][1] - fatias[i][0]) * W for i in range(size)
    ], dtype=np.int32)
    displacements = np.array([
        fatias[i][0] * W for i in range(size)
    ], dtype=np.int32)

    tempos = []

    # -----------------------------------------------------------------------
    # Warm-up (1 execução fora da medição)
    # -----------------------------------------------------------------------
    bloco_halo = image_full[halo_top:halo_bot, :].copy()
    resultado_local = _aplicar_filtro_bloco(bloco_halo, has_top_halo, has_bot_halo, nome_filtro)

    # Buffer de saída no rank 0: array PLANO e CONTÍGUO de H*W bytes.
    # Usar um array plano (1-D) como recvbuf garante que o Gatherv escreva
    # diretamente nos bytes corretos sem precisar de reshape extra.
    saida_flat = None
    if rank == 0:
        saida_flat = np.empty(H * W, dtype=np.uint8)

    # --- Warm-up (1 execução fora da medição) ---
    sendbuf_warm = np.ascontiguousarray(resultado_local.flatten())
    comm.Gatherv(
        sendbuf=sendbuf_warm,
        recvbuf=(saida_flat, send_counts, displacements, MPI.UNSIGNED_CHAR) if rank == 0 else None,
        root=0,
    )
    if rank == 0:
        log(f"[PAR] Warm-up concluído – filtro {nome_filtro}")

    # -----------------------------------------------------------------------
    # Loop de benchmark (30 repetições)
    # -----------------------------------------------------------------------
    for it in range(N_REPETICOES):
        # Barreira antes: todos os processos começam no mesmo ponto
        comm.Barrier()
        t0 = time.perf_counter()

        # --- Processamento local (cada processo trabalha na sua fatia) ---
        bloco_halo    = image_full[halo_top:halo_bot, :].copy()
        resultado_loc = _aplicar_filtro_bloco(
            bloco_halo, has_top_halo, has_bot_halo, nome_filtro
        )

        # --- Coleta no rank 0 via Gatherv (buffer numpy contíguo) ---
        sendbuf = np.ascontiguousarray(resultado_loc.flatten())
        comm.Gatherv(
            sendbuf=sendbuf,
            recvbuf=(saida_flat, send_counts, displacements, MPI.UNSIGNED_CHAR) if rank == 0 else None,
            root=0,
        )

        # Barreira depois: aguarda todos terminarem antes de parar o cronômetro
        comm.Barrier()
        t1 = time.perf_counter()

        t_local = t1 - t0
        # Tempo real da iteração = máximo entre todos os processos
        t_iter = comm.reduce(t_local, op=MPI.MAX, root=0)
        if rank == 0:
            tempos.append(t_iter)

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Reconstrói o array 2-D a partir do buffer plano
        saida_2d = saida_flat.reshape(H, W).copy()
        # Restaura bordas externas com os pixels originais (não processadas pelo filtro)
        saida_2d[0, :]   = image_full[0, :]
        saida_2d[H-1, :] = image_full[H-1, :]
        saida_2d[:, 0]   = image_full[:, 0]
        saida_2d[:, W-1] = image_full[:, W-1]
        fname = os.path.join(OUTPUT_DIR, f"par_{size}proc_{nome_filtro}.png")
        cv2.imwrite(fname, saida_2d)
        log(f"[PAR] Imagem salva: {fname}")

    return tempos if rank == 0 else []


# ===========================================================================
# Ponto de entrada principal
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # 1. Rank 0 carrega/gera a imagem
    # -----------------------------------------------------------------------
    image     = None
    img_shape = None
    img_src   = ""

    if rank == 0:
        image, img_src = carregar_ou_gerar_imagem()
        img_shape = image.shape  # (H, W)
        log(f"Imagem pronta: {img_shape[0]}x{img_shape[1]}")

    # Distribui o shape (apenas 2 inteiros – bcast de objeto pequeno é OK)
    img_shape = comm.bcast(img_shape, root=0)
    img_src   = comm.bcast(img_src,   root=0)

    H, W = img_shape

    # -----------------------------------------------------------------------
    # 2. Todos os processos alocam o buffer e recebem a imagem via Bcast
    #    (buffer numpy – sem pickle, operação eficiente)
    # -----------------------------------------------------------------------
    if rank != 0:
        image = np.empty((H, W), dtype=np.uint8)
    # comm.Bcast com buffer numpy: envia os bytes brutos do array
    comm.Bcast(image, root=0)
    log(f"Imagem recebida via Bcast ({H}x{W}={H*W/1e6:.2f} MB)")

    # Dicionário acumulador de resultados: { filtro: { n_proc: stats } }
    # Neste script, n_proc = size atual (o usuário rodará com -n 1, 2, 4, 8)
    # O CSV e o relatório serão gerados para cada execução separadamente.
    results_esta_execucao = {}

    # -----------------------------------------------------------------------
    # 3. Benchmark sequencial (apenas quando rodando com 1 processo OU
    #    para obter o baseline; aqui executamos sempre no rank 0 como ref.)
    # -----------------------------------------------------------------------
    seq_tempos = {}
    if rank == 0:
        log("=" * 60)
        log("BENCHMARK SEQUENCIAL")
        log("=" * 60)
        seq_raw = benchmark_sequencial(image)
        for nome in ("mean", "median"):
            stats = compute_stats(seq_raw[nome])
            seq_tempos[nome] = stats
            log(f"[SEQ] {nome}: média={stats['mean']:.4f}s  "
                f"std={stats['std']:.4f}s  n={stats['n']}")
            results_esta_execucao.setdefault(nome, {})[1] = stats

    # Compartilha tempos sequenciais com todos (para calcular speedup)
    seq_tempos = comm.bcast(seq_tempos, root=0)

    # -----------------------------------------------------------------------
    # 4. Benchmark paralelo
    # -----------------------------------------------------------------------
    if rank == 0:
        log("=" * 60)
        log(f"BENCHMARK PARALELO  ({size} processos)")
        log("=" * 60)

    for nome_filtro in ("mean", "median"):
        comm.Barrier()  # sincroniza antes de cada filtro
        if rank == 0:
            log(f"Iniciando benchmark paralelo – filtro {nome_filtro}...")

        tempos_par = benchmark_paralelo_filtro(image, nome_filtro)

        if rank == 0:
            stats_par = compute_stats(tempos_par)
            speedup   = compute_speedup(seq_tempos[nome_filtro]["mean"],
                                        stats_par["mean"])
            log(f"[PAR/{size}proc] {nome_filtro}: "
                f"média={stats_par['mean']:.4f}s  "
                f"std={stats_par['std']:.4f}s  "
                f"speedup={speedup:.3f}x  "
                f"n={stats_par['n']}")
            results_esta_execucao.setdefault(nome_filtro, {})[size] = stats_par

    # -----------------------------------------------------------------------
    # 5. Rank 0 salva resultados
    # -----------------------------------------------------------------------
    if rank == 0:
        log("=" * 60)
        log("Salvando resultados...")

        # CSV
        save_csv(results_esta_execucao, OUTPUT_DIR)

        # Gráficos
        plot_time_vs_procs(results_esta_execucao, OUTPUT_DIR)
        plot_speedup(results_esta_execucao, OUTPUT_DIR)

        # Relatório
        gerar_relatorio(results_esta_execucao, img_shape, img_src, OUTPUT_DIR)

        log("=" * 60)
        log("CONCLUÍDO. Verifique a pasta 'output/' para os resultados.")
        log("=" * 60)

        # Resumo no terminal
        print("\n" + "=" * 60)
        print(f"  RESUMO – {size} processo(s)")
        print("=" * 60)
        for filtro, proc_data in results_esta_execucao.items():
            nome = "Média" if filtro == "mean" else "Mediana"
            print(f"\n  Filtro {nome}:")
            seq_m = proc_data.get(1, {}).get("mean", None)
            for n_proc in sorted(proc_data.keys()):
                s = proc_data[n_proc]
                sp = 1.0 if n_proc == 1 else (
                    compute_speedup(seq_m, s["mean"]) if seq_m else 0.0
                )
                label = "Sequencial" if n_proc == 1 else f"{n_proc} processos"
                print(f"    {label:<15}: {s['mean']:.4f}s ± {s['std']:.4f}s  "
                      f"speedup={sp:.3f}x")
        print()


if __name__ == "__main__":
    main()

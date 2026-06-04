import os
import time
import json
import numpy as np
import cv2

from mpi4py import MPI

from filters import (
    aplicar_filtro_media_rapido,
    aplicar_filtro_mediana_rapido,
    calcular_bloco_media,
    calcular_bloco_mediana,
)

from benchmark import (
    calcular_estatisticas,
    calcular_speedup,
    salvar_csv,
    gerar_grafico_tempo_vs_processos,
    gerar_grafico_speedup,
    gerar_relatorio,
)

N_REPETICOES = 30             #repetições por filtro por configuração
DIRETORIO_SAIDA = "output"    #pasta de saída
CAMINHO_IMAGEM = "entrada.jpg"
TAMANHO_IMAGEM_SINTETICA = (4000, 4000)  #altura x largura da imagem sintética

#Comunicador MPI
comunicador = MPI.COMM_WORLD
rank = comunicador.Get_rank()
num_processos = comunicador.Get_size()


def registrar_log(mensagem: str):
    #Imprime uma mensagem prefixada com o rank (apenas rank 0 ou com prefixo)
    print(f"[rank {rank}] {mensagem}", flush=True)


def carregar_ou_gerar_imagem() -> tuple[np.ndarray, str]:
#carrega primeiro a imagem de entrada e se der errado gera uma

    if os.path.exists(CAMINHO_IMAGEM):
        imagem = cv2.imread(CAMINHO_IMAGEM, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            registrar_log(f"AVISO: não foi possível ler '{CAMINHO_IMAGEM}'. Gerando imagem sintética.")
        else:
            registrar_log(f"Imagem carregada: '{CAMINHO_IMAGEM}' ({imagem.shape[0]}x{imagem.shape[1]})")
            return imagem, f"arquivo '{CAMINHO_IMAGEM}' lido do disco"
            
    #Gera imagem sintética com ruído
    altura, largura = TAMANHO_IMAGEM_SINTETICA
    registrar_log(f"Gerando imagem sintética {altura}x{largura} com ruído...")
    gerador_aleatorio = np.random.default_rng(seed=42)
    
    #Base: gradiente suave
    base = np.tile(np.linspace(0, 255, largura, dtype=np.float32), (altura, 1))
    
    #Ruído gaussiano
    ruido = gerador_aleatorio.normal(0, 40, (altura, largura)).astype(np.float32)
    
    #Ruído impulsivo  – beneficia o filtro de mediana
    mascara_sal = gerador_aleatorio.random((altura, largura)) < 0.02
    mascara_pimenta = gerador_aleatorio.random((altura, largura)) < 0.02
    
    imagem = np.clip(base + ruido, 0, 255).astype(np.uint8)
    imagem[mascara_sal] = 255
    imagem[mascara_pimenta] = 0
    
    #Salva a imagem gerada para referência
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
    cv2.imwrite(os.path.join(DIRETORIO_SAIDA, "entrada_sintetica.png"), imagem)
    registrar_log(f"Imagem sintética salva em '{DIRETORIO_SAIDA}/entrada_sintetica.png'")
    
    return imagem, f"imagem sintética gerada ({altura}x{largura} pixels com ruído gaussiano e impulsivo)"


#Divide a imagem em fatias para o MPI
def calcular_fatias(altura: int, n_proc: int) -> list[tuple[int, int]]:
    #calcula os intervalos [inicio, fim) de linhas para cada processo

    base  = altura // n_proc
    resto = altura %  n_proc
    fatias = []
    inicio = 0
    for i in range(n_proc):
        # Processos 0..resto-1 recebem uma linha extra
        linhas = base + (1 if i < resto else 0)
        fatias.append((inicio, inicio + linhas))
        inicio += linhas
    return fatias


def benchmark_sequencial(imagem: np.ndarray) -> dict:
    #Executa o benchmark sequencial dos dois filtros no rank 0
    resultados = {"media": [], "mediana": []}
    funcoes_filtro = {
        "media":   aplicar_filtro_media_rapido,
        "mediana": aplicar_filtro_mediana_rapido,
    }

    for nome_filtro, funcao in funcoes_filtro.items():
        registrar_log(f"[SEQ] Aquecimento (Warm-up) – filtro {nome_filtro}...")
        _ = funcao(imagem)  #warm-up

        registrar_log(f"[SEQ] Benchmark – filtro {nome_filtro} ({N_REPETICOES} repetições)...")
        for it in range(N_REPETICOES):
            t0 = time.perf_counter()
            resultado = funcao(imagem)
            t1 = time.perf_counter()
            resultados[nome_filtro].append(t1 - t0)
            if it == N_REPETICOES - 1:
                #Salva a última imagem processada
                os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
                nome_arquivo = os.path.join(DIRETORIO_SAIDA, f"seq_{nome_filtro}.png")
                cv2.imwrite(nome_arquivo, resultado)
                registrar_log(f"[SEQ] Imagem salva: {nome_arquivo}")

    return resultados


#Paralela - um filtro por vez
def _aplicar_filtro_bloco(bloco_halo, possui_halo_topo, possui_halo_base, nome_filtro):
    if nome_filtro == "media":
        return calcular_bloco_media(bloco_halo, possui_halo_topo, possui_halo_base)
    else:
        return calcular_bloco_mediana(bloco_halo, possui_halo_topo, possui_halo_base)


def benchmark_paralelo_filtro(imagem_completa: np.ndarray, nome_filtro: str) -> list[float]:
    #Executa o benchmark paralelo de um filtro específico

    altura, largura = imagem_completa.shape
    fatias = calcular_fatias(altura, num_processos)  
    inicio_r, fim_r = fatias[rank]       
    n_linhas = fim_r - inicio_r          

    # Linhas de halo: 1 linha acima e 1 abaixo (se existirem)
    halo_topo = max(0, inicio_r - 1)
    halo_base = min(altura, fim_r + 1)
    possui_halo_topo = (halo_topo < inicio_r)
    possui_halo_base = (halo_base > fim_r)

    #Prepara estruturas para Gatherv
    contagens_envio = np.array([
        (fatias[i][1] - fatias[i][0]) * largura for i in range(num_processos)
    ], dtype=np.int32)
    
    deslocamentos = np.array([
        fatias[i][0] * largura for i in range(num_processos)
    ], dtype=np.int32)

    tempos = []

    #Warm-up (1 execução fora da medição)
    bloco_halo = imagem_completa[halo_topo:halo_base, :].copy()
    resultado_local = _aplicar_filtro_bloco(bloco_halo, possui_halo_topo, possui_halo_base, nome_filtro)

    #Buffer de saída no rank 0: array Plano e Contígio
    saida_plana = None
    if rank == 0:
        saida_plana = np.empty(altura * largura, dtype=np.uint8)

    #Sincronização do warm-up
    buffer_envio_warm = np.ascontiguousarray(resultado_local.flatten())
    comunicador.Gatherv(
        sendbuf=buffer_envio_warm,
        recvbuf=(saida_plana, contagens_envio, deslocamentos, MPI.UNSIGNED_CHAR) if rank == 0 else None,
        root=0,
    )
    if rank == 0:
        registrar_log(f"[PAR] Aquecimento concluído – filtro {nome_filtro}")

    #Loop de benchmark (30 repetições)
    for it in range(N_REPETICOES):
        #Barreira antes: todos os processos começam no mesmo ponto
        comunicador.Barrier()
        t0 = time.perf_counter()

        #Processamento local (cada processo trabalha na sua fatia)
        bloco_halo = imagem_completa[halo_topo:halo_base, :].copy()
        resultado_loc = _aplicar_filtro_bloco(
            bloco_halo, possui_halo_topo, possui_halo_base, nome_filtro
        )

        #Coleta no rank 0 via Gatherv (buffer numpy contíguo)
        buffer_envio = np.ascontiguousarray(resultado_loc.flatten())
        comunicador.Gatherv(
            sendbuf=buffer_envio,
            recvbuf=(saida_plana, contagens_envio, deslocamentos, MPI.UNSIGNED_CHAR) if rank == 0 else None,
            root=0,
        )

        #Barreira depois: aguarda todos terminarem antes de parar o cronômetro
        comunicador.Barrier()
        t1 = time.perf_counter()

        t_local = t1 - t0
        #Tempo real da iteração = máximo entre todos os processos
        t_iter = comunicador.reduce(t_local, op=MPI.MAX, root=0)
        if rank == 0:
            tempos.append(t_iter)

    if rank == 0:
        os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
        #Reconstrói a matriz 2-D a partir do buffer plano
        saida_2d = saida_plana.reshape(altura, largura).copy()
        
        #Restaura bordas externas com os pixels originais
        saida_2d[0, :] = imagem_completa[0, :]
        saida_2d[altura-1, :] = imagem_completa[altura-1, :]
        saida_2d[:, 0] = imagem_completa[:, 0]
        saida_2d[:, largura-1] = imagem_completa[:, largura-1]
        
        nome_arquivo = os.path.join(DIRETORIO_SAIDA, f"par_{num_processos}proc_{nome_filtro}.png")
        cv2.imwrite(nome_arquivo, saida_2d)
        registrar_log(f"[PAR] Imagem salva: {nome_arquivo}")

    return tempos if rank == 0 else []


def main():
    #Rank 0 carrega/gera a imagem
    imagem = None
    dimensoes_imagem = None
    origem_imagem = ""

    if rank == 0:
        imagem, origem_imagem = carregar_ou_gerar_imagem()
        dimensoes_imagem = imagem.shape  # (Altura, Largura)
        registrar_log(f"Imagem pronta: {dimensoes_imagem[0]}x{dimensoes_imagem[1]}")

    #Distribui as dimensões
    dimensoes_imagem = comunicador.bcast(dimensoes_imagem, root=0)
    origem_imagem = comunicador.bcast(origem_imagem, root=0)

    altura, largura = dimensoes_imagem

    #Todos os processos alocam o buffer e recebem a imagem via Bcast
    if rank != 0:
        imagem = np.empty((altura, largura), dtype=np.uint8)
        
    comunicador.Bcast(imagem, root=0)
    registrar_log(f"Imagem recebida via Bcast ({altura}x{largura}={altura*largura/1e6:.2f} MB)")

    resultados_esta_execucao = {}

    #Benchmark sequencial (linha de base)
    tempos_seq = {}
    if rank == 0:
        registrar_log("=" * 60)
        registrar_log("BENCHMARK SEQUENCIAL")
        registrar_log("=" * 60)
        seq_bruto = benchmark_sequencial(imagem)
        
        for nome_filtro in ("media", "mediana"):
            estatisticas = calcular_estatisticas(seq_bruto[nome_filtro])
            tempos_seq[nome_filtro] = estatisticas
            registrar_log(f"[SEQ] {nome_filtro}: média={estatisticas['media']:.4f}s  "
                          f"desvio_padrao={estatisticas['desvio_padrao']:.4f}s  n={estatisticas['n']}")
            resultados_esta_execucao.setdefault(nome_filtro, {})[1] = estatisticas

    #Compartilha tempos sequenciais com todos
    tempos_seq = comunicador.bcast(tempos_seq, root=0)

    #Benchmark paralelo
    if rank == 0:
        registrar_log("=" * 60)
        registrar_log(f"BENCHMARK PARALELO  ({num_processos} processos)")
        registrar_log("=" * 60)

    for nome_filtro in ("media", "mediana"):
        comunicador.Barrier()
        if rank == 0:
            registrar_log(f"Iniciando benchmark paralelo – filtro {nome_filtro}...")

        tempos_paralelos = benchmark_paralelo_filtro(imagem, nome_filtro)

        if rank == 0:
            estatisticas_par = calcular_estatisticas(tempos_paralelos)
            speedup = calcular_speedup(tempos_seq[nome_filtro]["media"], estatisticas_par["media"])
            
            registrar_log(f"[PAR/{num_processos}proc] {nome_filtro}: "
                          f"média={estatisticas_par['media']:.4f}s  "
                          f"desvio_padrao={estatisticas_par['desvio_padrao']:.4f}s  "
                          f"speedup={speedup:.3f}x  "
                          f"n={estatisticas_par['n']}")
                          
            resultados_esta_execucao.setdefault(nome_filtro, {})[num_processos] = estatisticas_par

    #Rank 0 salva resultados (com acúmulo de execuções)
    if rank == 0:
        registrar_log("=" * 60)
        registrar_log("Salvando resultados...")

        caminho_historico = os.path.join(DIRETORIO_SAIDA, "historico.json")
        todos_resultados = {}
        
        if os.path.exists(caminho_historico):
            with open(caminho_historico, "r") as f:
                todos_resultados = json.load(f)

        for filtro, dados_proc in resultados_esta_execucao.items():
            if filtro not in todos_resultados:
                todos_resultados[filtro] = {}
            for n_proc, estatisticas in dados_proc.items():
                todos_resultados[filtro][str(n_proc)] = estatisticas

        with open(caminho_historico, "w") as f:
            json.dump(todos_resultados, f)

        resultados_finais = {}
        for filtro, dados_proc in todos_resultados.items():
            resultados_finais[filtro] = {int(k): v for k, v in dados_proc.items()}

        salvar_csv(resultados_finais, DIRETORIO_SAIDA)
        gerar_grafico_tempo_vs_processos(resultados_finais, DIRETORIO_SAIDA)
        gerar_grafico_speedup(resultados_finais, DIRETORIO_SAIDA)
        gerar_relatorio(resultados_finais, dimensoes_imagem, origem_imagem, DIRETORIO_SAIDA)

        registrar_log("=" * 60)
        registrar_log("CONCLUÍDO. Verifique a pasta 'output/' para os resultados.")
        registrar_log("=" * 60)

        print("\n" + "=" * 60)
        print(f"  RESUMO – {num_processos} processo(s)")
        print("=" * 60)
        for filtro, dados_proc in resultados_esta_execucao.items():
            nome = "Média" if filtro == "media" else "Mediana"
            print(f"\n  Filtro {nome}:")
            media_seq = dados_proc.get(1, {}).get("media", None)
            for n_proc in sorted(dados_proc.keys()):
                est = dados_proc[n_proc]
                sp = 1.0 if n_proc == 1 else (
                    calcular_speedup(media_seq, est["media"]) if media_seq else 0.0
                )
                rotulo = "Sequencial" if n_proc == 1 else f"{n_proc} processos"
                print(f"    {rotulo:<15}: {est['media']:.4f}s ± {est['desvio_padrao']:.4f}s  "
                      f"speedup={sp:.3f}x")
        print()

if __name__ == "__main__":
    main()
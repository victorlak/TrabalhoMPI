"""
filters.py
==========
Implementações manuais dos filtros de média e mediana 3x3.
Nenhuma função do OpenCV é usada para aplicar os filtros – o cálculo é feito
explicitamente com numpy e laços, evidenciando o processamento.

Funções exportadas:
    - apply_mean_filter(image)       → versão sequencial completa
    - apply_median_filter(image)     → versão sequencial completa
    - compute_mean_block(block)      → aplica média em um bloco com halo
    - compute_median_block(block)    → aplica mediana em um bloco com halo

O "bloco com halo" é uma fatia da imagem que inclui 1 linha extra acima e
1 linha extra abaixo da região que será efetivamente processada, permitindo
que o filtro 3x3 seja aplicado corretamente nas fronteiras entre processos MPI.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Filtro de Média 3x3 – versão sequencial (imagem completa)
# ---------------------------------------------------------------------------

def apply_mean_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro de média 3x3 em toda a imagem (versão sequencial).

    Para cada pixel (i, j) que não está na borda externa da imagem, calcula
    a média aritmética dos 9 vizinhos na janela 3x3 centrada em (i, j).
    Os pixels da borda externa (primeira/última linha e coluna) são mantidos
    inalterados (copiados para a saída).

    Parâmetros
    ----------
    image : np.ndarray, shape (H, W), dtype uint8
        Imagem em escala de cinza.

    Retorna
    -------
    np.ndarray, shape (H, W), dtype uint8
        Imagem filtrada.
    """
    H, W = image.shape
    # Cria a saída já preenchida com os valores originais (bordas serão mantidas)
    output = image.copy()

    # Itera apenas nos pixels internos (exclui primeira/última linha e coluna)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            # Extrai janela 3x3 centrada em (i, j)
            window = image[i - 1:i + 2, j - 1:j + 2]
            # Média dos 9 pixels, arredondada para o inteiro mais próximo
            output[i, j] = np.uint8(np.round(window.mean()))

    return output


# ---------------------------------------------------------------------------
# Filtro de Mediana 3x3 – versão sequencial (imagem completa)
# ---------------------------------------------------------------------------

def apply_median_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro de mediana 3x3 em toda a imagem (versão sequencial).

    Para cada pixel (i, j) interno, ordena os 9 valores da janela 3x3 e
    seleciona o valor central (mediana). Os pixels da borda são mantidos.

    Parâmetros
    ----------
    image : np.ndarray, shape (H, W), dtype uint8

    Retorna
    -------
    np.ndarray, shape (H, W), dtype uint8
    """
    H, W = image.shape
    output = image.copy()

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            window = image[i - 1:i + 2, j - 1:j + 2]
            # np.median retorna float; converte para uint8
            output[i, j] = np.uint8(np.median(window))

    return output


# ---------------------------------------------------------------------------
# Versões vetorizadas mais rápidas (usadas pelo benchmark sequencial)
# ---------------------------------------------------------------------------

def apply_mean_filter_fast(image: np.ndarray) -> np.ndarray:
    """
    Filtro de média 3x3 vetorizado com numpy (para benchmark sequencial).

    Usa slicing numpy para construir a soma dos 9 vizinhos sem laço explícito
    em Python, tornando a execução muito mais rápida. Ainda é uma implementação
    manual – não usa cv2.blur ou funções similares.
    """
    H, W = image.shape
    img = image.astype(np.float32)
    output = image.copy().astype(np.float32)

    # Soma dos 9 vizinhos por slicing (equivalente à convolução 3x3 com kernel de uns)
    soma = (
        img[0:H-2, 0:W-2] + img[0:H-2, 1:W-1] + img[0:H-2, 2:W] +
        img[1:H-1, 0:W-2] + img[1:H-1, 1:W-1] + img[1:H-1, 2:W] +
        img[2:H,   0:W-2] + img[2:H,   1:W-1] + img[2:H,   2:W]
    )
    output[1:H-1, 1:W-1] = np.round(soma / 9.0)
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_median_filter_fast(image: np.ndarray) -> np.ndarray:
    """
    Filtro de mediana 3x3 vetorizado com numpy (para benchmark sequencial).

    Empilha as 9 janelas deslocadas em uma terceira dimensão e aplica
    np.median ao longo desse eixo. Ainda é implementação manual.
    """
    H, W = image.shape
    img = image.astype(np.float32)
    output = image.copy().astype(np.float32)

    # Empilha os 9 vizinhos como camadas de um array 3-D
    neighbors = np.stack([
        img[0:H-2, 0:W-2], img[0:H-2, 1:W-1], img[0:H-2, 2:W],
        img[1:H-1, 0:W-2], img[1:H-1, 1:W-1], img[1:H-1, 2:W],
        img[2:H,   0:W-2], img[2:H,   1:W-1], img[2:H,   2:W],
    ], axis=0)  # shape: (9, H-2, W-2)

    output[1:H-1, 1:W-1] = np.median(neighbors, axis=0)
    return np.clip(output, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Funções auxiliares para a versão paralela (processamento de blocos com halo)
# ---------------------------------------------------------------------------

def compute_mean_block(block_with_halo: np.ndarray, has_top_halo: bool,
                       has_bottom_halo: bool) -> np.ndarray:
    """
    Aplica o filtro de média 3x3 em um bloco que inclui linhas de halo.

    O "halo" são linhas extras recebidas dos processos vizinhos para que o
    filtro 3x3 nas fronteiras do bloco seja calculado corretamente.

    Parâmetros
    ----------
    block_with_halo : np.ndarray, shape (H_halo, W)
        Bloco da imagem incluindo as linhas de halo (se existirem).
    has_top_halo : bool
        True se a primeira linha do bloco é uma linha de halo (não pertence
        a este processo – não deve ser incluída na saída).
    has_bottom_halo : bool
        True se a última linha do bloco é uma linha de halo.

    Retorna
    -------
    np.ndarray
        Apenas as linhas que pertencem a este processo, após a filtragem.
    """
    H, W = block_with_halo.shape
    img = block_with_halo.astype(np.float32)
    output = block_with_halo.copy().astype(np.float32)

    # Linhas que podem ser computadas (têm vizinhos acima e abaixo dentro do bloco)
    for i in range(1, H - 1):
        soma = (
            img[i-1, 0:W-2] + img[i-1, 1:W-1] + img[i-1, 2:W] +
            img[i,   0:W-2] + img[i,   1:W-1] + img[i,   2:W] +
            img[i+1, 0:W-2] + img[i+1, 1:W-1] + img[i+1, 2:W]
        )
        output[i, 1:W-1] = np.round(soma / 9.0)

    # Remove as linhas de halo para retornar apenas o resultado desta fatia
    start = 1 if has_top_halo else 0
    end   = H - 1 if has_bottom_halo else H
    return np.clip(output[start:end], 0, 255).astype(np.uint8)


def compute_median_block(block_with_halo: np.ndarray, has_top_halo: bool,
                         has_bottom_halo: bool) -> np.ndarray:
    """
    Aplica o filtro de mediana 3x3 em um bloco com linhas de halo.

    Parâmetros e retorno análogos a compute_mean_block.
    """
    H, W = block_with_halo.shape
    img = block_with_halo.astype(np.float32)
    output = block_with_halo.copy().astype(np.float32)

    for i in range(1, H - 1):
        neighbors = np.stack([
            img[i-1, 0:W-2], img[i-1, 1:W-1], img[i-1, 2:W],
            img[i,   0:W-2], img[i,   1:W-1], img[i,   2:W],
            img[i+1, 0:W-2], img[i+1, 1:W-1], img[i+1, 2:W],
        ], axis=0)
        output[i, 1:W-1] = np.median(neighbors, axis=0)

    start = 1 if has_top_halo else 0
    end   = H - 1 if has_bottom_halo else H
    return np.clip(output[start:end], 0, 255).astype(np.uint8)

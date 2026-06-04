import numpy as np


#Filtro de Média 3x3: versão sequencial (imagem completa)
def aplicar_filtro_media(imagem: np.ndarray) -> np.ndarray:
    altura, largura = imagem.shape
    #Cria a saída já preenchida com os valores originais (bordas serão mantidas)
    saida = imagem.copy()

    #Itera apenas nos pixels internos (exclui primeira/última linha e coluna)
    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            #Extrai janela 3x3 centrada em (i, j)
            janela = imagem[i - 1:i + 2, j - 1:j + 2]
            #Média dos 9 pixels, arredondada para o inteiro mais próximo
            saida[i, j] = np.uint8(np.round(janela.mean()))

    return saida


#Filtro de Mediana 3x3: versão sequencial (imagem completa)
def aplicar_filtro_mediana(imagem: np.ndarray) -> np.ndarray:
    altura, largura = imagem.shape
    saida = imagem.copy()

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            janela = imagem[i - 1:i + 2, j - 1:j + 2]
            #np.median retorna float, converte para uint8
            saida[i, j] = np.uint8(np.median(janela))

    return saida


#Versões vetorizadas mais rápidas (usadas pelo benchmark sequencial)

def aplicar_filtro_media_rapido(imagem: np.ndarray) -> np.ndarray:
    altura, largura = imagem.shape
    img_float = imagem.astype(np.float32)
    saida = imagem.copy().astype(np.float32)

    #Soma dos 9 vizinhos por fatiamento (equivalente à convolução 3x3 com kernel de uns)
    soma = (
        img_float[0:altura-2, 0:largura-2] + img_float[0:altura-2, 1:largura-1] + img_float[0:altura-2, 2:largura] +
        img_float[1:altura-1, 0:largura-2] + img_float[1:altura-1, 1:largura-1] + img_float[1:altura-1, 2:largura] +
        img_float[2:altura,   0:largura-2] + img_float[2:altura,   1:largura-1] + img_float[2:altura,   2:largura]
    )
    saida[1:altura-1, 1:largura-1] = np.round(soma / 9.0)
    return np.clip(saida, 0, 255).astype(np.uint8)


def aplicar_filtro_mediana_rapido(imagem: np.ndarray) -> np.ndarray:
    altura, largura = imagem.shape
    img_float = imagem.astype(np.float32)
    saida = imagem.copy().astype(np.float32)

    #Empilha os 9 vizinhos como camadas de um array 3-D
    vizinhos = np.stack([
        img_float[0:altura-2, 0:largura-2], img_float[0:altura-2, 1:largura-1], img_float[0:altura-2, 2:largura],
        img_float[1:altura-1, 0:largura-2], img_float[1:altura-1, 1:largura-1], img_float[1:altura-1, 2:largura],
        img_float[2:altura,   0:largura-2], img_float[2:altura,   1:largura-1], img_float[2:altura,   2:largura],
    ], axis=0)  #dimensões: (9, altura-2, largura-2)

    saida[1:altura-1, 1:largura-1] = np.median(vizinhos, axis=0)
    return np.clip(saida, 0, 255).astype(np.uint8)


# Funções auxiliares para a versão paralela (processamento de blocos com halo)

def calcular_bloco_media(bloco_com_halo: np.ndarray, possui_halo_topo: bool,
                         possui_halo_base: bool) -> np.ndarray:
    altura, largura = bloco_com_halo.shape
    img_float = bloco_com_halo.astype(np.float32)
    saida = bloco_com_halo.copy().astype(np.float32)

    #Linhas que podem ser computadas (têm vizinhos acima e abaixo dentro do bloco)
    for i in range(1, altura - 1):
        soma = (
            img_float[i-1, 0:largura-2] + img_float[i-1, 1:largura-1] + img_float[i-1, 2:largura] +
            img_float[i,   0:largura-2] + img_float[i,   1:largura-1] + img_float[i,   2:largura] +
            img_float[i+1, 0:largura-2] + img_float[i+1, 1:largura-1] + img_float[i+1, 2:largura]
        )
        saida[i, 1:largura-1] = np.round(soma / 9.0)

    #Remove as linhas de halo para retornar apenas o resultado desta fatia
    inicio = 1 if possui_halo_topo else 0
    fim    = altura - 1 if possui_halo_base else altura
    return np.clip(saida[inicio:fim], 0, 255).astype(np.uint8)


def calcular_bloco_mediana(bloco_com_halo: np.ndarray, possui_halo_topo: bool,
                           possui_halo_base: bool) -> np.ndarray:
    altura, largura = bloco_com_halo.shape
    img_float = bloco_com_halo.astype(np.float32)
    saida = bloco_com_halo.copy().astype(np.float32)

    for i in range(1, altura - 1):
        vizinhos = np.stack([
            img_float[i-1, 0:largura-2], img_float[i-1, 1:largura-1], img_float[i-1, 2:largura],
            img_float[i,   0:largura-2], img_float[i,   1:largura-1], img_float[i,   2:largura],
            img_float[i+1, 0:largura-2], img_float[i+1, 1:largura-1], img_float[i+1, 2:largura],
        ], axis=0)
        saida[i, 1:largura-1] = np.median(vizinhos, axis=0)

    inicio = 1 if possui_halo_topo else 0
    fim    = altura - 1 if possui_halo_base else altura
    return np.clip(saida[inicio:fim], 0, 255).astype(np.uint8)
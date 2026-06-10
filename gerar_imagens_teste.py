import cv2
import numpy as np
import os

OUTPUT_DIR = "."
SIZE = 500

def img1_geometrica():
    """Formas geométricas grandes e nítidas: círculo, retângulo, triângulo."""
    img = np.ones((SIZE, SIZE), dtype=np.uint8) * 200  # fundo cinza claro

    # Retângulo preto no canto superior esquerdo
    img[30:170, 30:170] = 0

    # Círculo branco no centro
    cv2.circle(img, (SIZE//2, SIZE//2), 100, 255, -1)

    # Triângulo preto (via polígono)
    pts = np.array([[370, 30], [470, 170], [270, 170]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)

    # Retângulo branco no canto inferior direito
    img[330:470, 330:470] = 255

    # Círculo preto pequeno no centro
    cv2.circle(img, (SIZE//2, SIZE//2), 30, 0, -1)

    # Adicionar texto nítido
    cv2.putText(img, "TESTE", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 60, 2)
    cv2.putText(img, "123", (220, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    # Borda preta fina
    img[:2, :] = 0
    img[-2:, :] = 0
    img[:, :2] = 0
    img[:, -2:] = 0

    return img


def img2_checkerboard():
    """Checkerboard + barras + texto, alto contraste."""
    img = np.ones((SIZE, SIZE), dtype=np.uint8) * 255

    # Tabuleiro de xadrez 8x8
    cell = SIZE // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = 0

    # Faixas horizontais nas bordas superior/inferior
    img[0:10, :] = 128
    img[SIZE-10:SIZE, :] = 128

    # Faixas verticais nas laterais
    img[:, 0:10] = 128
    img[:, SIZE-10:SIZE] = 128

    # Números grandes em alguns quadrados (claros)
    cv2.putText(img, "A", (55, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 3)
    cv2.putText(img, "B", (180, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 3)
    cv2.putText(img, "C", (305, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 3)
    cv2.putText(img, "D", (55, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 3)
    cv2.putText(img, "E", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 3)
    cv2.putText(img, "F", (305, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 3)

    return img


def img3_gradiente_animeis():
    """Gradiente suave + retângulos nítidos + ruído sal e pimenta pesado."""
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)

    # Gradiente vertical suave (fundo)
    for y in range(SIZE):
        val = int(255 * y / SIZE)
        img[y, :] = val

    # Retângulos de bordas nítidas (preto, branco, cinza médio)
    img[30:120, 30:120] = 0
    img[150:240, 30:120] = 255
    img[270:360, 30:120] = 128

    # Círculos concêntricos
    for r in range(50, 200, 30):
        cv2.circle(img, (400, 300), r, 255 if (r//30) % 2 == 0 else 0, 4)

    # Ruído sal e pimenta pesado (para testar mediana)
    rng = np.random.default_rng(42)
    salt = rng.random((SIZE, SIZE)) < 0.05
    pepper = rng.random((SIZE, SIZE)) < 0.05
    img[salt] = 255
    img[pepper] = 0

    # Borda clara
    img[:3, :] = 255
    img[-3:, :] = 255
    img[:, :3] = 255
    img[:, -3:] = 255

    return img


if __name__ == "__main__":
    images = {
        "entrada_geometrica.jpg": img1_geometrica(),
        "entrada_checkerboard.jpg": img2_checkerboard(),
        "entrada_gradiente.jpg": img3_gradiente_animeis(),
    }

    for fname, img in images.items():
        path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(path, img)
        print(f"Salva: {path} ({img.shape[0]}x{img.shape[1]})")

    print("\nImagens geradas! Escolha uma e renomeie para 'entrada.jpg' para usar.")
    print("Exemplo: cp entrada_geometrica.jpg entrada.jpg")

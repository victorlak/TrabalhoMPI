# Comparação de Filtros de Suavização com MPI

## Descrição
Este projeto implementa e compara os filtros de suavização de imagem (média 3x3 e mediana 3x3) nas versões sequencial e paralela com MPI, utilizando Python e `mpi4py`. Este repositório contém a versão principal do trabalho. Detalhes específicos e implementações individuais podem ser encontrados nas *branches* de cada integrante do grupo.

## Estrutura do Projeto

* `main.py` – Script principal MPI (ponto de entrada).
* `filters.py` – Implementação manual dos filtros (sem funções prontas como `cv2.blur`).
* `benchmark.py` – Funções de benchmark, estatísticas e geração de relatório dinâmico.
* `requirements.txt` – Dependências Python do projeto.
* `entrada.jpg` – *(Opcional)* Imagem de entrada. Se ausente, uma imagem sintética gigante (4000x4000) com ruído será gerada automaticamente para forçar o processamento.
* `output/` – Pasta criada automaticamente na execução, contendo:
  * `entrada_sintetica.png` – Imagem sintética gerada (se aplicável).
  * `seq_media.png` / `seq_mediana.png` – Resultados da execução sequencial.
  * `par_Nproc_media.png` / `par_Nproc_mediana.png` – Resultados paralelos por número de processos.
  * `historico.json` – Arquivo invisível que acumula os dados de múltiplas execuções.
  * `resultados_benchmark.csv` – Tabela final de resultados em CSV.
  * `tempo_vs_processos.png` / `speedup_vs_processos.png` – Gráficos automáticos.
  * `relatorio.txt` – Relatório textual completo gerado na última execução.

---

## Pré-requisitos e Instalação

É necessário ter Python 3.10+ e uma implementação do MPI instalada no sistema. Recomenda-se fortemente o uso de um ambiente virtual (`venv`).

### Instalação no Linux (Debian / Ubuntu)
1. Instale o OpenMPI e os cabeçalhos de desenvolvimento no sistema:
   ```bash
   sudo apt update
   sudo apt install openmpi-bin libopenmpi-dev python3-dev
   ```
2. Crie e ative o ambiente virtual:
```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Instale o mpi4py compilando-o do zero (para evitar erros de biblioteca dinâmica libmpi.so) e as demais dependências:
```bash
   pip install --no-binary mpi4py mpi4py
   pip install numpy opencv-python matplotlib
   ```

### Instalação no Windows
1. Baixe o Microsoft MPI (MS-MPI) acessando a página de releases (https://github.com/microsoft/Microsoft-MPI/releases).
2. Baixe e instale ambos os arquivos: msmpisetup.exe (runtime) e msmpisdk.msi (SDK).
3. Reinicie o seu terminal (ou o computador) para aplicar as variáveis de ambiente.
4. Crie o ambiente virtual, ative e instale as dependências:
```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Como Executar

O programa foi atualizado para acumular os resultados. Para gerar os gráficos comparativos completos e a tabela do relatório, você deve rodar os comandos em sequência, variando o número de processos (-n).

Execute a partir da pasta raiz do projeto com o ambiente virtual ativado:
```bash
# 1. Executa a versão sequencial e o baseline (1 processo)
mpiexec -n 1 python main.py

# 2. Executa a versão paralela com 2 processos
mpiexec -n 2 python main.py

# 3. Executa a versão paralela com 4 processos
mpiexec -n 4 python main.py

# 4. Executa a versão paralela com 8 processos
mpiexec -n 8 python main.py
   ```

**Aviso para usuários Linux:** Se o seu processador tiver menos de 8 núcleos físicos, o OpenMPI bloqueará a execução do último comando alegando falta de "slots". Para contornar isso e forçar a execução, adicione a flag de sobreposição:

mpiexec --oversubscribe -n 8 python main.py
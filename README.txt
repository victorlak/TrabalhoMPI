================================================================================
  README – Comparação de Filtros de Suavização com MPI
================================================================================

DESCRIÇÃO
---------
Este projeto implementa e compara os filtros de suavização de imagem
(média 3x3 e mediana 3x3) nas versões sequencial e paralela com MPI,
usando Python + mpi4py.

ESTRUTURA DO PROJETO
---------------------
  main.py          – Script principal MPI (ponto de entrada)
  filters.py       – Implementação manual dos filtros (sem cv2.blur/medianBlur)
  benchmark.py     – Funções de benchmark, estatísticas e geração de relatório
  requirements.txt – Dependências Python
  README.txt       – Este arquivo
  entrada.jpg      – (opcional) Imagem de entrada; se ausente, uma imagem
                     sintética 4000x4000 com ruído é gerada automaticamente.
  output/          – Pasta criada automaticamente com:
    entrada_sintetica.png      – Imagem sintética gerada (se aplicável)
    seq_mean.png               – Resultado sequencial do filtro de média
    seq_median.png             – Resultado sequencial do filtro de mediana
    par_Nproc_mean.png         – Resultado paralelo do filtro de média (N proc.)
    par_Nproc_median.png       – Resultado paralelo do filtro de mediana
    benchmark_results.csv      – Tabela de resultados em CSV
    tempo_vs_processos.png     – Gráfico de tempo por nº de processos
    speedup_vs_processos.png   – Gráfico de speedup
    relatorio.txt              – Relatório completo gerado automaticamente

PRÉ-REQUISITOS
--------------
1. Python 3.10 ou superior
2. MPI instalado no sistema:
     - Windows : Microsoft MPI (MS-MPI) – https://github.com/microsoft/Microsoft-MPI
     - Linux   : OpenMPI  (sudo apt install libopenmpi-dev openmpi-bin)
     - macOS   : OpenMPI  (brew install open-mpi)
3. Dependências Python (instale com pip):

     pip install -r requirements.txt

   OU instale individualmente:

     pip install mpi4py numpy opencv-python matplotlib

INSTALAÇÃO DO MS-MPI (Windows)
--------------------------------
1. Acesse: https://github.com/microsoft/Microsoft-MPI/releases
2. Baixe e instale msmpisetup.exe (runtime) E msmpisdk.msi (SDK).
3. Reinicie o terminal após a instalação.
4. Verifique: mpiexec --version

EXECUÇÃO
--------
Execute a partir da pasta do projeto (onde estão os arquivos .py):

  # Versão com 1 processo (referência sequencial):
  mpiexec -n 1 python main.py

  # Versão com 2 processos:
  mpiexec -n 2 python main.py

  # Versão com 4 processos:
  mpiexec -n 4 python main.py

  # Versão com 8 processos:
  mpiexec -n 8 python main.py

OBSERVAÇÕES DE EXECUÇÃO
------------------------
• Cada execução gera um arquivo benchmark_results.csv e relatorio.txt
  independente na pasta output/. Execute com diferentes valores de -n
  para coletar os dados de todas as configurações.

• A imagem "entrada.jpg" é opcional. Se não existir, o programa gera
  automaticamente uma imagem sintética 4000x4000 pixels com ruído
  gaussiano e ruído impulsivo ("sal e pimenta").

• Para usar sua própria imagem, copie-a para a pasta do projeto com o
  nome "entrada.jpg". Imagens maiores (≥ 1000x1000) são recomendadas
  para observar ganho de desempenho com MPI.

• O benchmark executa 30 repetições por filtro por configuração, com
  1 warm-up antes das medições. Outliers são removidos pelo método IQR
  antes de calcular as estatísticas finais.

• Em máquinas com menos de 8 núcleos físicos, o speedup com 8 processos
  pode ser menor do que com 4 (overhead de escalonamento do SO).

EXEMPLO DE SAÍDA NO TERMINAL
------------------------------
  [rank 0] Imagem pronta: 4000x4000
  [rank 0] Imagem recebida via Bcast (4000x4000=16.00 MB)
  [rank 0] ============================================================
  [rank 0] BENCHMARK SEQUENCIAL
  [rank 0] ============================================================
  [rank 0] [SEQ] Warm-up – filtro mean...
  [rank 0] [SEQ] Benchmark – filtro mean (30 repetições)...
  ...
  [rank 0] CONCLUÍDO. Verifique a pasta 'output/' para os resultados.

SOLUÇÃO DE PROBLEMAS
---------------------
• "ModuleNotFoundError: No module named 'mpi4py'":
    pip install mpi4py

• "mpiexec não é reconhecido":
    Instale o MS-MPI (Windows) ou OpenMPI (Linux/macOS) conforme
    as instruções acima.

• Execução lenta com 8 processos:
    Normal em máquinas com poucos núcleos. O overhead de comunicação
    MPI pode superar o ganho de paralelismo para máquinas com ≤ 4 núcleos.

• Imagem muito pequena → pouco speedup:
    Use uma imagem maior ou deixe o programa gerar a sintética 4000x4000.

AUTORES / DISCIPLINA
---------------------
Atividade Avaliativa – Programação Paralela com MPI
Filtros de Suavização de Imagem: Média 3x3 e Mediana 3x3

================================================================================

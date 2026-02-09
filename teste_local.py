import imm_module
import time
import os

# --- CONFIGURAÇÃO ---
# Caminhos relativos ou absolutos para os arquivos gerados pelo gerador_deterministico.py
arquivo_grafo = r"C:\Users\hugor\OneDrive\Desktop\Nova pasta (2)\grafo_cross_platform.txt"
arquivo_universo = r"C:\Users\hugor\OneDrive\Desktop\Nova pasta (2)\universe.txt"

# Verificação básica
if not os.path.exists(arquivo_grafo):
    print(f"ERRO: Grafo não encontrado: {arquivo_grafo}")
    print("Rode o gerador_deterministico.py primeiro.")
    exit()

print("==================================================")
print(" INICIANDO PONTE PYTHON -> C++ ")
print("==================================================")
print(f"Grafo: {arquivo_grafo}")
if os.path.exists(arquivo_universo):
    print(f"Universo Restrito: {arquivo_universo}")
else:
    print("Universo: NÃO DETECTADO (Busca Global)")

inicio = time.time()

# Chamada atualizada
imm_module.run_cplusplus(
    arquivo_grafo, 
    50,             # k (sementes)
    "tvic",         # modelo 
    0.5,            # epsilon
    10000,          # numMC
    arquivo_universo if os.path.exists(arquivo_universo) else "" # Passa o universo ou string vazia
)

fim = time.time()

print("==================================================")
print(f" SUCESSO! Tempo Total: {fim - inicio:.4f} s")
print("==================================================")
import imm_module
import time

arquivo_grafo = r"C:\Users\hugor\OneDrive\Desktop\Nova pasta (2)\grafo_cross_platform.txt"

# Exemplo: Sementes obtidas via algoritmo de PageRank ou Degree Centrality no NetworkX
sementes_heuristica = [10, 45, 89, 102, 500] 

print(f"Avaliando {len(sementes_heuristica)} sementes customizadas...")
inicio = time.time()

# Retorna diretamente um float com a média de influência
spread_medio = imm_module.evaluate_seeds(
    arquivo_grafo, 
    sementes_heuristica, 
    "tvic", 
    10000
)

fim = time.time()

print(f"Spread Médio Estimado: {spread_medio:.2f}")
print(f"Tempo da Simulação: {fim - inicio:.4f} s")
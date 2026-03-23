import imm_module

arquivo_grafo = r"C:\Users\hugor\OneDrive\Desktop\Nova pasta (2)\grafo_cross_platform.txt"

# Um dicionário agrupando suas abordagens
meus_conjuntos = {
    "Degree Centrality": [10, 45, 89, 102, 500],
    "PageRank": [12, 18, 99, 305, 412],
    "Aleatório": [5, 55, 120, 900, 1002]
}

print("Iniciando bateria de testes...")

# Retorna um dicionário com os spreads calculados
resultados = imm_module.evaluate_seeds(
    arquivo_grafo, 
    meus_conjuntos, 
    "tvic", 
    10000
)

print("\n--- Resumo dos Resultados ---")
for metodo, spread in resultados.items():
    print(f"{metodo}: {spread:.2f}")
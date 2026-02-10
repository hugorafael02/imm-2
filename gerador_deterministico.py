import numpy as np
import random
import sys
import time

def generate_deterministic_scale_free(
    num_nodes, 
    edges_per_node, 
    seed, 
    output_file, 
    universe_output_file, # Novo parâmetro
    universe_percentage,  # Novo parâmetro (0.0 a 1.0)
    alpha=1, 
    beta_param=5
):
    print(f"--- Configuração Blindada (Cross-Platform) ---")
    print(f"Nós: {num_nodes:,}")
    print(f"Arestas/Nó (m): {edges_per_node}")
    print(f"Semente: {seed}")
    
    # 1. TRAVA AS SEMENTES
    # Importante: O estado do random determina a forma do grafo.
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    # ==============================================================================
    # PARTE 1: GERAÇÃO DO GRAFO (Lógica Original Intocada)
    # ==============================================================================
    print(f"\n[1/2] Gerando Grafo em '{output_file}'...")
    
    with open(output_file, 'w', newline='\n') as f:
        
        # Urna de preferential attachment
        targets = list(range(edges_per_node)) * edges_per_node
        
        buffer = []
        BUFFER_SIZE = 50000 
        
        # Gera arestas iniciais
        for i in range(edges_per_node):
            for j in range(i + 1, edges_per_node):
                # Peso aleatório (Beta distribution)
                w = np.random.beta(alpha, beta_param)
                buffer.append(f"{i} {j} {w:.6f}\n")

        # Loop principal de Preferential Attachment
        for source in range(edges_per_node, num_nodes):
            
            # Seleciona m vizinhos existentes baseados em grau (targets)
            neighbors = set()
            while len(neighbors) < edges_per_node:
                idx = random.randint(0, len(targets) - 1)
                neighbor = targets[idx]
                neighbors.add(neighbor)
            
            # Ordenação para determinismo cross-platform
            for neighbor in sorted(neighbors):
                weight = np.random.beta(alpha, beta_param)
                buffer.append(f"{source} {neighbor} {weight:.6f}\n")
                
                targets.append(source)
                targets.append(neighbor)
            
            # Buffer de escrita
            if len(buffer) >= BUFFER_SIZE:
                f.writelines(buffer)
                buffer = []
                
                if source % 100000 == 0:
                    perc = (source / num_nodes) * 100
                    print(f"\rProgresso Grafo: {perc:.1f}% | Nós: {source:,}", end="")

        if buffer:
            f.writelines(buffer)

    print(f"\nGrafo concluído.")

    # ==============================================================================
    # PARTE 2: GERAÇÃO DO UNIVERSO ADMISSÍVEL (Nova Feature)
    # ==============================================================================
    print(f"\n[2/2] Gerando Universo Admissível ({universe_percentage*100:.1f}%) em '{universe_output_file}'...")
    
    # Calculamos quantos nós farão parte do universo
    universe_size = int(num_nodes * universe_percentage)
    
    # O random.seed já foi "gastado" pela geração do grafo.
    # Para garantir que o universo seja determinístico INDEPENDENTEMENTE do tamanho do grafo,
    # ou para manter consistência, podemos resetar a seed ou continuar a sequência.
    # Aqui, opto por continuar a sequência (já é determinístico pois depende da seed inicial),
    # mas usamos random.sample que é eficiente.
    
    # Criamos o universo sorteando 'universe_size' números únicos do intervalo [0, num_nodes)
    # random.sample é perfeito para isso (sem reposição).
    admissible_nodes = random.sample(range(num_nodes), universe_size)
    
    # Ordenamos para ficar bonito no arquivo e facilitar leitura humana/debug
    admissible_nodes.sort()
    
    with open(universe_output_file, 'w', newline='\n') as f_uni:
        # Usando bufferização manual simples (join é muito rápido em Python)
        chunk_size = 50000
        for i in range(0, len(admissible_nodes), chunk_size):
            chunk = admissible_nodes[i : i + chunk_size]
            # Escreve um ID por linha
            f_uni.write('\n'.join(map(str, chunk)) + '\n')

    total_time = time.time() - start_time
    print(f"Sucesso Total! Arquivos gerados.")
    print(f" - Grafo: {output_file}")
    print(f" - Universo: {universe_output_file} ({universe_size:,} nós)")
    print(f"Tempo Total: {total_time:.2f}s")

# --- EXECUÇÃO ---
if __name__ == "__main__":
    generate_deterministic_scale_free(
        num_nodes=1_000_000_0,     
        edges_per_node=10,       
        seed=424242,             
        output_file="grafo_cross_platform.txt",
        
        # Novos Parâmetros
        universe_output_file="universe.txt",
        universe_percentage=0.2, # 20% dos nós serão selecionados
        
        alpha=1, 
        beta_param=3
    )
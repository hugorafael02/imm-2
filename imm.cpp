#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <utility> // Para std::move e std::make_pair
#include <map>
#include <omp.h>

// Headers de C para compatibilidade total (necessários para Linux/GCC)
#include <cstdio>  // Para printf
#include <cstdlib> // Para atoi, atof, exit, system

#include "./utils.h"

using namespace std;

vector<int> imm(int V, vector<vector<edge> >& rs, string model, int k, double eps, double ell,
	const vector<bool>& candidates);
int greedy(int V, vector<vector<int> > &h2v, vector<vector<int> > &v2h, int k,
		vector<int> &S);


// ============================================================================
// MONTE CARLO IC - VERSÃO ESTÁVEL (SAFE)
// ============================================================================
int MonteCarlo_IC(int V, vector<vector<edge> >& es, vector<int>& S, mt19937& gen) {

	queue<int> Q;
	int count = 0;
	uniform_real_distribution<> dis_prob(0.0, 1.0);

	// ALOCAÇÃO SEGURA (LOCAL)
	// Para V=75k, isso é minúsculo (9KB). Não vai gargalar seu PC.
	// O sistema operacional gerencia isso automaticamente, sem risco de "sujeira" de execuções anteriores.
	vector<bool> active(V, false);

	for (int s : S) {
		// Proteção extra: ignora sementes inválidas
		if (s < V && !active[s]) {
			active[s] = true;
			Q.push(s);
			count++;
		}
	}

	while (!Q.empty()) {
		int u = Q.front();
		Q.pop();

		for (auto& e : es[u]) {
			int v = e.v;

			// Proteção de limites (Bound Check implícito pela lógica, mas garantido aqui)
			if (v < V && !active[v]) {
				if (dis_prob(gen) <= e.c) {
					active[v] = true;
					Q.push(v);
					count++;
				}
			}
		}
	}
	return count;
}

// ============================================================================
// MONTE CARLO LT - VERSÃO ESTÁVEL (SAFE)
// ============================================================================
int MonteCarlo_LT(int V, vector<vector<edge> >& es, vector<int>& S, mt19937& gen) {

	queue<int> Q;
	int count = 0;
	uniform_real_distribution<> dis_prob(0.0, 1.0);

	// ALOCAÇÃO SEGURA
	vector<bool> active(V, false);
	vector<double> incoming_weights(V, 0.0);
	vector<double> thresholds(V, -1.0); // -1 indica não gerado

	for (int s : S) {
		if (s < V && !active[s]) {
			active[s] = true;
			Q.push(s);
			count++;
		}
	}

	while (!Q.empty()) {
		int u = Q.front();
		Q.pop();

		for (auto& e : es[u]) {
			int v = e.v;

			if (v >= V) continue; // Proteção contra nós fora do limite
			if (active[v]) continue;

			// Lazy Threshold
			if (thresholds[v] < 0) {
				thresholds[v] = dis_prob(gen);
			}

			incoming_weights[v] += e.c;

			if (incoming_weights[v] >= thresholds[v]) {
				active[v] = true;
				Q.push(v);
				count++;
			}
		}
	}
	return count;
}


vector<int> gen_RR_IC(int V, vector<vector<edge> >& rs, mt19937& gen) {
	uniform_int_distribution<> dis_node(0, V - 1);
	uniform_real_distribution<> dis_prob(0.0, 1.0);
	
	vector<int> RR;
	queue<int> Q;

	// USAMOS UM SET PARA NÃO ALOCAR VETOR GIGANTE DE 4MB A CADA ITERAÇÃO
	unordered_set<int> visited;

	int z = dis_node(gen);

	visited.insert(z); // Marca visitado
	RR.push_back(z);
	Q.push(z);

	while (!Q.empty()) {
		int u = Q.front();
		Q.pop();

		for (auto& e : rs[u]) {
			int v = e.u;

			// Se NÃO está no set (count == 0), então não foi visitado
			if (visited.count(v) == 0) {
				if (dis_prob(gen) <= e.c) {
					visited.insert(v);
					RR.push_back(v);
					Q.push(v);
				}
			}
		}
	}
	return RR;
}

// GERAÇÃO DE RR-SETS PARA LINEAR THRESHOLD (LT)
// Lógica: Random Walk Reverso (Caminhada Aleatória)
vector<int> gen_RR_LT(int V, vector<vector<edge> >& rs, mt19937& gen) {
	uniform_int_distribution<> dis_node(0, V - 1);
	uniform_real_distribution<> dis_prob(0.0, 1.0);

	vector<int> RR;

	// Otimização: Set para não alocar memória O(V)
	unordered_set<int> visited;

	// 1. Seleciona a Raiz
	int u = dis_node(gen);
	RR.push_back(u);
	visited.insert(u);

	// 2. Loop de Caminhada (Random Walk)
	// Diferente do IC, não precisamos de fila (Queue), pois o caminho é linear.
	// Só seguimos UM antecessor por vez.
	while (true) {

		// Se não há vizinhos de entrada, a caminhada morre
		if (rs[u].empty()) {
			break;
		}

		// 3. Roleta Russa (Weighted Selection)
		// Sorteamos um número entre 0 e 1
		double val = dis_prob(gen);
		double cumulative = 0.0;
		int next_node = -1;

		// Percorre os vizinhos acumulando os pesos (e.p)
		for (auto& e : rs[u]) {
			cumulative += e.c;
			if (val <= cumulative) {
				next_node = e.u; // Escolheu este vizinho
				break;
			}
		}

		// 4. Transição
		// Se selecionamos alguém E ele ainda não foi visitado (evita ciclos)
		if (next_node != -1 && visited.count(next_node) == 0) {
			visited.insert(next_node);
			RR.push_back(next_node);
			u = next_node; // Avança para o próximo nó
		}
		else {
			// Se a roleta caiu numa faixa "vazia" (val > soma dos pesos)
			// OU se encontramos um ciclo -> Paramos.
			break;
		}
	}

	return RR;
}

// --- GREEDY OTIMIZADO (CELF / LAZY) COM FILTRO DE UNIVERSO ---
int greedy(int V, vector<vector<int> >& h2v, vector<vector<int> >& v2h, int k, 
           vector<int>& S, const vector<bool>& candidates) { // <--- NOVO PARAMETRO
	int H = (int)h2v.size();
	vector<bool> dead(H, false);
	vector<int> deg(V);

	// Fila de Prioridade para CELF
	priority_queue<pair<int, int> > Q;

	// 1. Inicialização
	for (int v = 0; v < V; v++) {
        // --- ALTERAÇÃO AQUI ---
        // Se o nó não faz parte do universo admissível, ignoramos ele.
        if (!candidates[v]) continue; 
        // ----------------------

		deg[v] = (int)v2h[v].size();
		if (deg[v] > 0) {
			Q.push(make_pair(deg[v], v));
		}
	}

	int total_covered = 0;
	vector<bool> selected(V, false);

	// 2. Loop Lazy (O resto permanece idêntico)
	while (S.size() < k && !Q.empty()) {
		pair<int, int> top = Q.top();
		Q.pop();

		int v = top.second;
		int stored_deg = top.first;

		if (selected[v]) continue;

		if (stored_deg == deg[v]) {
			S.push_back(v);
			selected[v] = true;
			total_covered += deg[v];

			for (int h_idx : v2h[v]) {
				if (!dead[h_idx]) {
					dead[h_idx] = true;
					for (int u : h2v[h_idx]) {
						deg[u]--;
					}
				}
			}
		}
		else {
			Q.push(make_pair(deg[v], v));
		}
	}
	return total_covered;
}

// ... (outros includes permanecem os mesmos)

// =================================================================================
// FUNÇÃO AUXILIAR: GERAÇÃO E INDEXAÇÃO DE RR-SETS
// Faz o trabalho pesado: Gera 'needed' amostras, salva em h2v e atualiza v2h
// =================================================================================
void generate_samples(int V, vector<vector<edge> >& rs, string model, long long needed,
	vector<vector<int> >& h2v, vector<vector<int> >& v2h,
	long long& totW, int seed_salt) {

	if (needed <= 0) return;

	int initial_H_size = h2v.size();

	// 1. GERAÇÃO PARALELA (Com buffer local e std::move)
#pragma omp parallel
	{
		vector<vector<int> > local_h2v;

		unsigned long seed = (unsigned long)(time(NULL) ^ (omp_get_thread_num() * 12345 + seed_salt));
		mt19937 gen(seed);

#pragma omp for nowait
		for (long long j = 0; j < needed; j++) {
			vector<int> RR;
			if (model == "tvlt") {
				RR = gen_RR_LT(V, rs, gen);
			}
			else {
				// Fallback
				RR = gen_RR_IC(V, rs, gen);
			}
			local_h2v.push_back(RR);
		}

		// Merge Global Otimizado
#pragma omp critical
		{
			for (auto& rr : local_h2v) {
				h2v.push_back(std::move(rr));
			}
		}
	}

	// 2. INDEXAÇÃO SEQUENCIAL (v2h)
	// Processa apenas o que foi adicionado agora (do initial_H_size em diante)
	int final_H_size = h2v.size();
	for (int idx = initial_H_size; idx < final_H_size; idx++) {
		for (int v : h2v[idx]) {
			v2h[v].push_back(idx);
			totW++;
		}
	}
}

// =================================================================================
// FUNÇÃO PRINCIPAL IMM
// =================================================================================
vector<int> imm(int V, vector<vector<edge> >& rs, string model, int k, double eps, double ell,
	const vector<bool>& candidates) {
	const double e = exp(1);
	double log_VCk = log_nCk(V, k);

	ell = ell * (1 + log(2) / log(V));
	double eps_p = sqrt(2) * eps;

	printf("ell  = %f\n", ell);
	printf("eps' = %f\n", eps_p);
	printf("log{V c k} = %f\n", log_VCk);

	double OPT_lb = 1;

	int H = 0;
	vector<vector<int> > h2v;
	vector<vector<int> > v2h(V);
	long long int totW = 0;

	// --- FASE 1: Estimação ---
	for (int i = 1; i <= log2(V) - 1; i++) {
		double x = V / pow(2, i);
		double lambda_prime = (2 + 2.0 / 3.0 * eps_p)
			* (log_VCk + ell * log(V) + log(log2(V))) * V / (eps_p * eps_p);
		double theta_i = lambda_prime / x;

		printf("i = %d\n", i);
		printf("x  = %.0f\n", x);
		printf("theta_i = %.0f\n", theta_i);

		long long iterations_needed = (long long)(theta_i - H);

		// CHAMADA DA FUNÇÃO AUXILIAR
		generate_samples(V, rs, model, iterations_needed, h2v, v2h, totW, i);

		H = h2v.size(); // Atualiza H real

		printf("H  = %d\n", H);
		printf("totW = %lld\n", totW);

		vector<int> S;
		int degS = greedy(V, h2v, v2h, k, S, candidates);
		printf("deg(S) = %d\n", degS);
		printf("Inf(S) = %f\n", 1.0 * V * degS / H);
		printf("\n");

		if (1.0 * V * degS / theta_i >= (1 + eps_p) * x) {
			OPT_lb = (1.0 * V * degS) / ((1 + eps_p) * theta_i);
			break;
		}
	}

	// --- CÁLCULO DE PARÂMETROS ---
	double lambda_star;
	{
		double alpha = sqrt(ell * log(V) + log(2));
		double beta = sqrt((1 - 1 / e) * (log_VCk + ell * log(V) + log(2)));
		double c = (1 - 1 / e) * alpha + beta;
		lambda_star = 2 * V * c * c / (eps * eps);
	}
	double theta = lambda_star / OPT_lb;
	printf("OPT_ = %.0f\n", OPT_lb);
	printf("lambda* = %.0f\n", lambda_star);
	printf("theta = %.0f\n", theta);

	// --- FASE 2: Refinamento ---
	long long iterations_needed_2 = (long long)(theta - H);

	// CHAMADA DA FUNÇÃO AUXILIAR
	// Passamos um salt fixo grande (ex: 67890) para diferenciar da Fase 1
	generate_samples(V, rs, model, iterations_needed_2, h2v, v2h, totW, 67890);

	H = h2v.size();
	printf("H  = %d\n", H);

	vector<int> S;
	int degS = greedy(V, h2v, v2h, k, S, candidates);
	printf("deg(S) = %d\n", degS);
	printf("Inf(S) = %f\n", 1.0 * V * degS / H);

	return S;
}

void run(map<string, string> args) {
	// ========================================================================
	// 1. PARSEAMENTO DE ARGUMENTOS E CONFIGURAÇÃO
	// ========================================================================
	// Extrai os parâmetros passados via linha de comando ou definidos no main.
	// get_or_die garante que a execução pare se algo essencial faltar.
	// ========================================================================
	string input = get_or_die(args, "graph");
	int k = atoi(get_or_die(args, "k").c_str());         // Tamanho do conjunto semente (Budget)
	double eps = atof(get_or_die(args, "eps").c_str());  // Epsilon: Parâmetro de erro (ex: 0.1 ou 0.5)
	double ell = atof(get_or_die(args, "ell").c_str());  // Ell: Parâmetro de confiança probabilística (1/n^ell)
	string model = get_or_die(args, "model");            // Modelo de difusão: IC (Independent Cascade) ou LT (Linear Threshold)
	int numMC = atoi(get_or_die(args, "numMC").c_str()); // Número de simulações para a validação final

	// ========================================================================
	// 2. LEITURA DO GRAFO (I/O)
	// ========================================================================
	cout << "[1/4] Lendo arquivo..." << endl;
	ifstream is(input.c_str());
	if (!is.is_open()) {
		cerr << "Erro fatal: Nao foi possivel abrir " << input << endl;
		exit(1);
	}

	// Vetor temporário para armazenar a lista de arestas bruta.
	// Usamos isso para calcular V (número de nós) dinamicamente antes de alocar o grafo final.
	vector<edge> ps;
	int V = 0;
	int u, v;
	double p_val;

	// Formato esperado: u v weight
	// u: nó origem, v: nó destino, weight: probabilidade de propagação
	while (is >> u >> v >> p_val) {
		// Ignora auto-loops (nó influenciando a si mesmo), pois não agrega ganho marginal.
		if (u == v) continue;

		edge e = { u, v, p_val };

		// O número de nós V é o maior índice encontrado + 1 (assumindo índices base-0)
		V = max(V, max(u, v) + 1);
		ps.push_back(e);
	}
	is.close();

	// ========================================================================
	// 3. EXIBIÇÃO DE METADADOS
	// ========================================================================
	cout << "========================================" << endl;
	cout << "       IMM - CONFIGURACAO ATUAL" << endl;
	cout << "========================================" << endl;
	cout << "Dataset      : " << input << endl;
	cout << "Nos (V)      : " << V << endl;
	cout << "Arestas (E)  : " << ps.size() << endl;
	cout << "----------------------------------------" << endl;
	cout << "Modelo       : " << (model == "tvlt" ? "Linear Threshold (LT)" : "Independent Cascade (IC)") << endl;
	cout << "Sementes (k) : " << k << endl;
	cout << "Monte Carlo  : " << numMC << " simulacoes (Validacao)" << endl;
	cout << "========================================" << endl << endl;

	// ========================================================================
	// 4. CONSTRUÇÃO DAS ESTRUTURAS DO GRAFO
	// ========================================================================
	cout << "[2/4] Construindo grafo (V=" << V << ", E=" << ps.size() << ")..." << endl;

	// ========================================================================
	// NOVO BLOCO: CONFIGURAÇÃO DO UNIVERSO ADMISSÍVEL
	// ========================================================================
	vector<bool> candidates(V, true); // Por padrão, TODOS são candidatos
	string universe_file = "";

	if (args.count("universe")) {
		universe_file = args["universe"];
		cout << "[Info] Arquivo de universo detectado: " << universe_file << endl;

		ifstream u_file(universe_file.c_str());
		if (u_file.is_open()) {
			// Se abriu, primeiro setamos tudo como false
			fill(candidates.begin(), candidates.end(), false);

			int node_id;
			int count_valid = 0;
			while (u_file >> node_id) {
				if (node_id >= 0 && node_id < V) {
					candidates[node_id] = true;
					count_valid++;
				}
			}
			u_file.close();
			cout << "[Info] Universo restrito a " << count_valid << " nos." << endl;
		}
		else {
			cout << "[Aviso] Arquivo de universo nao encontrado. Usando todos os nos." << endl;
		}
	}
	else {
		cout << "[Info] Nenhum universo especificado. Busca global." << endl;
	}


	// rs (Reverse Structure): Grafo Transposto (v -> u). 
	// CRÍTICO PARA O IMM: Usado para gerar os RR-Sets (caminhando "para trás" a partir de um alvo).
	vector<vector<edge> > rs(V);

	// es (Edge Structure): Grafo Direto (u -> v).
	// Usado apenas na Validação Monte Carlo (propagação "para frente").
	vector<vector<edge> > es(V);

	for (auto e : ps) {
		rs[e.v].push_back(e); // Aresta entra em v vindo de u (útil para RR-Set)
		es[e.u].push_back(e); // Aresta sai de u indo para v (útil para Monte Carlo)
	}

	// --- TRUQUE DE MEMÓRIA (SWAP TRICK) ---
	// O vetor 'ps' (lista bruta) não é mais necessário e ocupa muita RAM (ex: grafos de 80M arestas).
	// .clear() não libera memória imediatamente, apenas marca como vazio.
	// .swap() com um vetor vazio força a desalocação imediata da memória do heap.
	{
		vector<edge> empty;
		ps.swap(empty);
	}

	// ========================================================================
	// 5. EXECUÇÃO DO ALGORITMO IMM (CORE)
	// ========================================================================
	cout << "[3/4] Executando IMM (Selecao de Sementes)..." << endl;
	clock_t start_imm = clock();

	// Chama o algoritmo principal. 
	// Retorna S: vetor contendo os IDs dos k nós mais influentes.
	vector<int> S = imm(V, rs, model, k, eps, ell, candidates);

	clock_t end_imm = clock();

	cout << "\n>>> TEMPO IMM: " << (double)(end_imm - start_imm) / CLOCKS_PER_SEC << "s <<<" << endl;
	cout << "Sementes Escolhidas: { ";
	for (size_t i = 0; i < S.size(); i++) cout << S[i] << (i < S.size() - 1 ? ", " : "");
	cout << " }" << endl;

	// ========================================================================
	// 6. VALIDAÇÃO VIA MONTE CARLO (PÓS-PROCESSAMENTO)
	// ========================================================================
	// O IMM dá uma garantia teórica (1 - 1/e). Aqui rodamos simulações reais
	// para medir a propagação "verdadeira" empírica do conjunto S encontrado.
	// ========================================================================
	cout << "\n[4/4] Validacao Monte Carlo (" << numMC << " simulacoes)..." << endl;

	clock_t start_mc = clock();

	// Variáveis acumuladoras globais (compartilhadas entre threads)
	double total_inf = 0.0;             // Soma total da influência (para média final)
	double global_running_spread = 0.0; // Soma parcial (apenas para exibição em tempo real)
	int progress_counter = 0;           // Contador de simulações concluídas

	// Início da Região Paralela (OpenMP)
	/*
	-------------------------------------------------------------------------------
    Cria um time de threads (ex: 8 threads se houver 8 núcleos). 
    Tudo dentro do bloco { ... } é executado por TODAS as threads simultaneamente.
    É aqui que as variáveis locais da thread são alocadas (ex: seeds de random).
	-------------------------------------------------------------------------------
	*/
#pragma omp parallel
	{
		// Geração de semente aleatória única por thread.
		// Se usássemos apenas time(NULL), todas as threads teriam a mesma semente,
		// gerando resultados idênticos e enviesando a validação.
		unsigned long seed = (unsigned long)(time(NULL) ^ (omp_get_thread_num() * 99999));
		mt19937 gen(seed);

		double local_inf = 0; // Acumulador local da thread (evita contenção de escrita)

		// Loop paralelo dinâmico
		/*
		-------------------------------------------------------------------------------
		Distribui as iterações de um loop entre as threads do time.
		Sem isso, todas as threads executariam o loop inteiro repetidamente.
		Ex: Thread 0 faz i=0 a 100, Thread 1 faz i=101 a 200...
		-------------------------------------------------------------------------------
		*/
#pragma omp for
		for (int sim = 0; sim < numMC; sim++) {
			int result = 0;

			// Simula a propagação usando o Grafo Direto (es)
			if (model == "tvic" || model == "ic") {
				result = MonteCarlo_IC(V, es, S, gen);
			}
			else if (model == "tvlt" || model == "lt") {
				result = MonteCarlo_LT(V, es, S, gen);
			}

			local_inf += result;

			// Atualização Atômica: Garante que duas threads não escrevam ao mesmo tempo
			// na variável de visualização (global_running_spread).

			/*
			-------------------------------------------------------------------------------
			[ A. #pragma omp atomic ] ("O Cirurgião")
			- O QUE É: Instrução de hardware para operações matemáticas simples.
			- ALVO: Uma única atualização de memória (x++, x += y).
			- PERFORMANCE: Altíssima (Nanosegundos). Quase sem custo extra.
			- USE PARA: Contadores globais, somas de estatísticas (ex: global_running_spread).

			[ B. #pragma omp critical ] ("O Guarda de Trânsito")
			- O QUE É: Um bloqueio de software (Mutex/Lock).
			- ALVO: Blocos inteiros de código (lógica complexa, I/O, containers).
			- PERFORMANCE: Lenta. Obriga as threads a fazerem fila (uma por vez).
			- USE PARA: Inserir em vetores (push_back), imprimir na tela (cout),
				   ou lógicas que não são apenas somas matemáticas.
			-------------------------------------------------------------------------------
			*/
#pragma omp atomic
			global_running_spread += result;

			// Barra de Progresso
#pragma omp atomic
			progress_counter++;

			// Exibe progresso a cada 100 iterações ou no final
			// "critical" garante que o cout não embaralhe texto de várias threads
			if (progress_counter % 100 == 0 || progress_counter == numMC) {
#pragma omp critical
				{
					double current_avg = global_running_spread / progress_counter;
					// \r retorna o cursor para o início da linha (efeito de animação)
					cout << "\rProgresso: " << progress_counter << "/" << numMC
						<< " (" << (int)(100.0 * progress_counter / numMC) << "%) "
						<< "| Spread Est.: " << fixed << setprecision(2) << current_avg << "   " << flush;
				}
			}
		}

		// Ao final do loop da thread, somamos o total local ao global de forma segura
#pragma omp atomic
		total_inf += local_inf;
	}
	// Fim da Região Paralela

	clock_t end_mc = clock();

	// ========================================================================
	// 7. RESULTADOS FINAIS
	// ========================================================================
	cout << "\n\nCalculo Finalizado." << endl;
	double final_spread = total_inf / numMC;

	cout << "========================================" << endl;
	cout << "SPREAD MEDIO FINAL: " << final_spread << endl;
	cout << "TEMPO MONTE CARLO : " << (double)(end_mc - start_mc) / CLOCKS_PER_SEC << "s" << endl;
	cout << "========================================" << endl;

	cout << "Pressione ENTER para sair..." << endl;
	cin.get();
}


int main() {
	// 1. CAMINHO DO ARQUIVO REAL
	// Usamos o Raw String Literal R"(...)" para o caminho que você definiu
	string graph_file = R"(C:\Users\hugor\OneDrive\Desktop\Nova pasta (2)\grafo_cross_platform.txt)";
	string universe_path = R"(C:\Users\hugor\OneDrive\Desktop\Nova pasta (2)\universe.txt)";

	// 2. PARÂMETROS DA EXECUÇÃO
	int k_seeds = 100;  // Vamos buscar os 50 nós mais influentes (tamanho padrão para papers)
	int sim_mc = 100000; // Número de simulações de Monte Carlo para validação (1000 é um bom balanço)

	// 3. PREPARAR ARGUMENTOS
	map<string, string> args;
	args["graph"] = graph_file;
	args["k"] = to_string(k_seeds);

	args["model"] = "tvic";          // Placeholder (nossa lógica ignora o tempo)
	args["eps"] = "0.5";             // Epsilon 0.1 é o padrão de mercado para precisão
	args["ell"] = "1.0";
	
	args["numMC"] = to_string(sim_mc);

	if (!universe_path.empty()) {
		args["universe"] = universe_path;
	}

	// Executa o algoritmo
	// A função run() vai chamar nossa leitura de 3 colunas e processar tudo.
	run(args);

	cout << "\n========================================" << endl;
	cout << "         FIM DA EXECUCAO" << endl;
	cout << "========================================" << endl;

	system("pause");
	return 0;
}
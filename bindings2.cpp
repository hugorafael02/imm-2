#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essencial: converte list do Python para std::vector do C++
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>

// Define a flag para desativar o main() dentro do imm.cpp
#define PYBIND_MODE

// Inclui o código fonte principal
#include "imm.cpp" 

namespace py = pybind11;

// Wrapper atualizado com o parametro universe_file
void run_algorithm(std::string graph_path, int k, std::string model, double eps, int numMC, std::string universe_file) {
    std::map<std::string, std::string> args;
    args["graph"] = graph_path;
    args["k"] = std::to_string(k);
    args["model"] = model;
    args["eps"] = std::to_string(eps);
    args["ell"] = "1.0";
    args["numMC"] = std::to_string(numMC);

    if (!universe_file.empty()) {
        args["universe"] = universe_file;
    }

    run(args);
}

// NOVO: Wrapper para rodar apenas o Monte Carlo passando uma lista de sementes
double evaluate_seeds_wrapper(std::string graph_path, std::vector<int> S, std::string model, int numMC) {
    // 1. Leitura do Grafo (focada apenas na estrutura direta 'es' para propagação)
    std::ifstream is(graph_path.c_str());
    if (!is.is_open()) {
        throw std::runtime_error("Erro fatal: Nao foi possivel abrir o arquivo do grafo.");
    }

    std::vector<edge> ps;
    int V = 0;
    int u, v;
    double p_val;

    while (is >> u >> v >> p_val) {
        if (u == v) continue;
        V = std::max(V, std::max(u, v) + 1);
        ps.push_back({u, v, p_val});
    }
    is.close();

    std::vector<std::vector<edge> > es(V);
    for (auto e : ps) {
        es[e.u].push_back(e);
    }
    
    // Libera a memória da lista bruta
    {
        std::vector<edge> empty;
        ps.swap(empty);
    }

    // 2. Simulação Monte Carlo Paralela (idêntica à do imm.cpp)
    double total_inf = 0.0;

    #pragma omp parallel
    {
        unsigned long seed = (unsigned long)(time(NULL) ^ (omp_get_thread_num() * 99999));
        std::mt19937 gen(seed);
        double local_inf = 0;

        #pragma omp for
        for (int sim = 0; sim < numMC; sim++) {
            if (model == "tvic" || model == "ic") {
                local_inf += MonteCarlo_IC(V, es, S, gen);
            }
            else if (model == "tvlt" || model == "lt") {
                local_inf += MonteCarlo_LT(V, es, S, gen);
            }
        }

        #pragma omp atomic
        total_inf += local_inf;
    }

    return total_inf / numMC;
}

PYBIND11_MODULE(imm_module, m) {
    m.doc() = "Modulo C++ Otimizado (Windows/Linux)";

    m.def("run_cplusplus", &run_algorithm,
        "Executa o IMM + Monte Carlo",
        py::arg("graph_path"),
        py::arg("k"),
        py::arg("model"),
        py::arg("eps"),
        py::arg("numMC"),
        py::arg("universe_file") = "");

    // NOVO: Expondo a função de avaliação independente
    m.def("evaluate_seeds", &evaluate_seeds_wrapper,
        "Avalia o spread de um conjunto customizado de sementes via Monte Carlo",
        py::arg("graph_path"),
        py::arg("seeds"), // Recebe uma lista Python, pybind converte para std::vector<int>
        py::arg("model"),
        py::arg("numMC"));
}
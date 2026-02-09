#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <map>
#include <string>
#include <iostream>

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

    // Só adiciona ao mapa se a string não for vazia
    if (!universe_file.empty()) {
        args["universe"] = universe_file;
    }

    run(args);
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
        py::arg("universe_file") = ""); // Valor padrão é string vazia (busca global)
}
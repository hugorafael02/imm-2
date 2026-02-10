#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>

struct edge {
    int u, v;
    double c;
};

// Implementação direta no Header (Header-only) permite melhor otimização

inline void init_args(int argc, char* argv[], std::map<std::string, std::string>& args) {
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a[0] != '-') continue;
        size_t at = a.find("=");
        if (at == std::string::npos) continue;
        std::string key = a.substr(1, at - 1);
        std::string val = a.substr(at + 1);
        args[key] = val;
    }
}

inline std::string get_or_die(std::map<std::string, std::string>& argv, std::string key) {
    if (argv.count(key) == 0) {
        std::cerr << "Erro: Chave '" << key << "' obrigatoria nao encontrada.\n";
        exit(1);
    }
    return argv[key];
}

inline double log_nCk(int n, int k) {
    if (k < 0 || k > n) return 0;
    // Usa lgamma para performance O(1) em vez de loop O(N)
    return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
}

#endif

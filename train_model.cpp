#include <iostream>
#include <array>
#include <cstdio>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include <string>

std::string executeCommand(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

int main() {
    const int num_threads = 4; // Número de hilos
    const char* commands[num_threads] = {
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\0.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\1.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\2.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\3.json"
    };

    std::vector<std::string> outputs(num_threads); // Vector para almacenar salidas
    std::vector<std::string> errors(num_threads); // Vector para almacenar errores

    double start_time = omp_get_wtime();

    // Paralelizar la ejecución de comandos
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < num_threads; ++i) {
        try {
            outputs[i] = executeCommand(commands[i]);
        } catch (const std::exception& e) {
            errors[i] = "Error in command " + std::to_string(i) + ": " + e.what();
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "Tiempo total de ejecución paralela: " << (end_time - start_time) << " segundos." << std::endl;

    // Imprimir resultados fuera del bloque crítico
    for (int i = 0; i < num_threads; ++i) {
        if (!outputs[i].empty()) {
            std::cout << "Output of command " << i << ": \n" << outputs[i] << std::endl;
        }
        if (!errors[i].empty()) {
            std::cerr << errors[i] << std::endl;
        }
    }

    return 0;
}

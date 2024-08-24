#include <iostream>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>
#include <omp.h> // Incluye OpenMP para medir el tiempo

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
    const int num_commands = 4; // Número de comandos
    const char* commands[num_commands] = {
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\0.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\1.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\2.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\3.json"
    };

    std::vector<std::string> outputs(num_commands); // Vector para almacenar salidas
    std::vector<std::string> errors(num_commands); // Vector para almacenar errores

    double start_time = omp_get_wtime(); // Comienza a medir el tiempo

    // Ejecución secuencial de los comandos
    for (int i = 0; i < num_commands; ++i) {
        try {
            outputs[i] = executeCommand(commands[i]);
        } catch (const std::exception& e) {
            errors[i] = "Error in command " + std::to_string(i) + ": " + e.what();
        }
    }

    double end_time = omp_get_wtime(); // Termina de medir el tiempo

    // Imprimir resultados
    for (int i = 0; i < num_commands; ++i) {
        if (!outputs[i].empty()) {
            std::cout << "Output of command " << i << ": \n" << outputs[i] << std::endl;
        }
        if (!errors[i].empty()) {
            std::cerr << errors[i] << std::endl;
        }
    }

    std::cout << "Tiempo total de ejecución secuencial: " << (end_time - start_time) << " segundos." << std::endl;

    return 0;
}

#include <iostream>    // Para la entrada y salida estándar
#include <array>       // Para manejar arreglos de tamaño fijo
#include <cstdio>      // Para funciones de C como popen
#include <omp.h>       // Para programación paralela con OpenMP
#include <stdexcept>   // Para manejo de excepciones estándar
#include <vector>      // Para usar vectores de la STL
#include <string>      // Para usar cadenas de la STL

// Función para ejecutar un comando del sistema y capturar su salida
std::string executeCommand(const std::string& command) {
    std::array<char, 128> buffer;  // Buffer para almacenar la salida del comando
    std::string result;            // Cadena para almacenar toda la salida concatenada
    FILE* pipe = popen(command.c_str(), "r");  // Ejecuta el comando en modo lectura
    if (!pipe) {
        throw std::runtime_error("popen() failed!");  // Lanza excepción si popen falla
    }
    // Lee la salida del comando línea por línea
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();  // Concatenar cada línea al resultado
    }
    pclose(pipe);  // Cierra el pipe
    return result;  // Retorna toda la salida del comando
}

int main() {
    const int num_threads = 4;  // Número de hilos para la ejecución paralela
    const char* commands[num_threads] = {
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\0.json",  // Comando 1
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\1.json",  // Comando 2
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\2.json",  // Comando 3
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\3.json"   // Comando 4
    };

    std::vector<std::string> outputs(num_threads);  // Vector para almacenar las salidas de cada comando
    std::vector<std::string> errors(num_threads);   // Vector para almacenar errores si ocurren

    double start_time = omp_get_wtime();  // Obtener el tiempo de inicio

    // Paralelizar la ejecución de comandos usando OpenMP
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < num_threads; ++i) {
        try {
            outputs[i] = executeCommand(commands[i]);  // Ejecutar cada comando y almacenar la salida
        } catch (const std::exception& e) {
            errors[i] = "Error in command " + std::to_string(i) + ": " + e.what();  // Capturar y almacenar errores
        }
    }

    double end_time = omp_get_wtime();  // Obtener el tiempo de finalización
    std::cout << "Tiempo total de ejecución paralela: " << (end_time - start_time) << " segundos." << std::endl;

    // Imprimir resultados fuera del bloque crítico para evitar condiciones de carrera
    for (int i = 0; i < num_threads; ++i) {
        if (!outputs[i].empty()) {
            std::cout << "Output of command " << i << ": \n" << outputs[i] << std::endl;  // Imprimir la salida del comando
        }
        if (!errors[i].empty()) {
            std::cerr << errors[i] << std::endl;  // Imprimir errores si los hay
        }
    }

    return 0;  // Indicar que el programa terminó correctamente
}

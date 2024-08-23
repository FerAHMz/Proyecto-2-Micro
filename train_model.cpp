#include <iostream>
#include <fstream>
#include <array>
#include <cstdio>
#include <omp.h>

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
    const int num_threads = 4;
    const char* commands[num_threads] = {
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\0.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\1.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\2.json",
        "python C:\\Proyecto-2-Micro\\train_model.py C:\\Proyecto-2-Micro\\3.json"
    };

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        try {
            std::string output = executeCommand(commands[i]);
            #pragma omp critical
            std::cout << "Output of command " << i << ": \n" << output << std::endl;
        } catch (const std::exception& e) {
            #pragma omp critical
            std::cerr << "Error in command " << i << ": " << e.what() << std::endl;
        }
    }

    return 0;
}

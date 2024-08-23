#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <omp.h>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main() {
    const int num_commands = 4;
    const char* commands[num_commands] = {
        "python train_model.py params1.json", 
        "python train_model.py params2.json",
        "python train_model.py params3.json",
        "python train_model.py params4.json"
    };

    #pragma omp parallel for num_threads(4)  
    for (int i = 0; i < num_commands; ++i) {
        try {
            std::string output = exec(commands[i]);
            #pragma omp critical  
            {
                std::cout << "Output of command " << i << ": \n" << output << std::endl;
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Error in command " << i << ": " << e.what() << std::endl;
            }
        }
    }

    return 0;
}


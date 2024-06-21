
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include <sstream>

#include <random>
#define TEST
using namespace std;

int64_t neg_mod(int64_t val, int64_t mod)
{
    return ((val % mod) + mod) % mod;
}

vector<vector<uint64_t>> read_data(const string &filename)
{
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;

    while (getline(input_file, line))
    {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ','))
        {
            row.push_back(stoll(cell));
        }

        data.push_back(row);
    }

    input_file.close();
    return data;
}

int main()
{

    string bolt_path = "/home/FASTST/data/BOLT/data.txt";
    string iron_path = "/home/FASTST/data/IRON/data.txt";
    vector<vector<uint64_t>> input_plain = read_data(bolt_path);

    uint64_t h1[2 * 6] = {0};
    int32_t bitlength = 32;
    uint64_t prime_mod = 4293918721; // 32 bit
    cout << "palin: " << "\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            cout << " " << input_plain[i][j] << " ";
        }
    }

    cout << "\n plain2: " << "\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 10; j++)
        {

            h1[i * 2 + j] = ((int64_t)input_plain[i][j]) >> 7;
            cout << " " << h1[i * 2 + j] << " ";
        }

        cout << "\n";
    }

    cout << "\n neg: " << "\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 10; j++)
        {

            h1[i * 2 + j] = neg_mod(((int64_t)input_plain[i][j]) >> 7, (int64_t)prime_mod);
            cout << " " << h1[i * 2 + j] << " ";
        }

        cout << "\n";
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(-200, 300);

    int randomNumber = distribution(gen);

    std::cout << "Random number between -200 and 300: " << randomNumber << std::endl;
}

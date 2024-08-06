#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

void load_dataset(vector<vector<double>> &datas, vector<int> &label, const string &filename)
{
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        istringstream record(line);
        vector<double> data;
        data.push_back(1.0);
        double temp;
        while (record >> temp)
            data.push_back(temp);
        label.push_back(int(temp));
        data.pop_back();
        datas.push_back(data);
    }
}

double scalarProduct(const vector<double> &w, const vector<double> &x)
{
    double ret = 0.0;
    for (int i = 0; i < w.size(); i++)
    {
        ret += w[i] * x[i];
    }
    return ret;
}

inline double sigmoid(const double &z)
{
    return 1 / (1 + exp(-z));
}

inline double sigmoid(const double &x)
{
}

vector<vector<double>> matTranspose(vector<vector<double>> &dataMat)
{
    vector<vector<double>> ret(dataMat[0].size(), vector<double>(dataMat.size(), 0));
    for (int i = 0; i < ret.size(); i++)
        for (int j = 0; j < ret[0].size(); j++)
            ret[i][j] = dataMat[j][i];
    return ret;
}

void gradAscent(vector<double> &weight,
                vector<vector<double>> &dataMat, vector<int> &labelMat, int maxCycles = 1000, double alpha = 0.01)
{
    const size_t data_size = dataMat.size();
    vector<vector<double>> dataMatT = matTranspose(dataMat);
    while (maxCycles > 0)
    {
        vector<double> h;
        vector<double> error;
        double sum_err = 0;
        for (auto &data : dataMat)
            h.push_back(sigmoid(scalarProduct(data, weight)));
        for (int i = 0; i < labelMat.size(); i++)
        {
            double dist = labelMat[i] - h[i];
            if (abs(dist) < 1e-10)
                dist = 0;
            error.push_back(dist);
        }
        for (int i = 0; i < weight.size(); i++)
            weight[i] += alpha * scalarProduct(dataMatT[i], error);
        double sum_error = 0.;
        for (int i = 0; i < data_size; ++i)
        {
            sum_error += -1 * labelMat[i] * log(h[i]) - (1 - labelMat[i]) * log(1 - h[i]);
        }
        printf("loss: %.10lf\n", sum_error / data_size);
        maxCycles--;
    }
}

inline int classify(vector<double> &data, vector<double> &weights)
{
    return sigmoid(scalarProduct(data, weights)) > 0.5 ? 1 : 0;
}

double testResult(vector<vector<double>> &testDataMat,
                  vector<int> &testDataLabel, vector<double> &weight)
{
    double errCount = 0.0;
    double dataSize = testDataMat.size();
    for (int i = 0; i < dataSize; i++)
        if (classify(testDataMat[i], weight) != testDataLabel[i])
            errCount += 1.0;
    return errCount / dataSize;
}

int main()
{
    vector<vector<double>> train_mat;
    vector<int> train_label;
    // string train_file("/data/PrivLR/ACAD");
    // string train_file("/data/PrivLR/HFCR");
    string train_file("/data/PrivLR/WIBC");
    load_dataset(train_mat, train_label, train_file);

    vector<vector<double>> test_mat;
    vector<int> test_label;
    // string test_file("/data/PrivLR/ACAD_test");
    // string test_file("/data/PrivLR/HFCR_test");
    string test_file("/data/PrivLR/WIBC_test");
    load_dataset(test_mat, test_label, test_file);

    vector<double> weight(train_mat[0].size(), 1);

    gradAscent(weight, train_mat, train_label, 10, 0.008);
    auto err = testResult(test_mat, test_label, weight);
    std::cout << "accuracy: " << (1 - err) * 100 << " %\n";
}
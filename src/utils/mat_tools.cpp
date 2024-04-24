#include "mat_tools.h"

std::vector<double> matmul(
    const std::vector<double> &mat1,
    const std::vector<double> &mat2, size_t dim1, size_t dim2, size_t dim3, bool trans)
{
    std::vector<double> result(dim1 * dim3);
    size_t i, j, k;
    if (!trans)
    {
        for (i = 0; i < dim1; i++)
        {
            for (j = 0; j < dim3; j++)
            {
                for (k = 0; k < dim2; k++)
                {
                    result[i * dim3 + j] += mat1[i * dim2 + k] * mat2[k * dim3 + j];
                }
            }
        }
    }
    else
    {
        for (i = 0; i < dim1; i++)
        {
            for (j = 0; j < dim3; j++)
            {
                for (k = 0; k < dim2; k++)
                {
                    result[i * dim3 + j] += mat1[i * dim2 + k] * mat2[j * dim3 + k];
                }
            }
        }
    }
    return result;
}

void random_mat(std::vector<double> &mat, double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    size_t size = mat.size();
    for (size_t i = 0; i < size; i++)
    {
        mat[i] = dist(gen);
    }
}

std::vector<double> zero_sum(size_t row, size_t column)
{
    std::vector<double> mat(row * column);
    random_mat(mat);
    size_t i, j;
    for (i = 0; i < row; i++)
    {
        double sum = 0.;
        for (j = 0; j < column - 1; j++)
        {
            sum += mat[i * column + j];
        }
        mat[(i + 1) * column - 1] = -sum;
    }
    return mat;
}

void load_mat(std::vector<double> &mat, const char *path)
{
    random_mat(mat);
}

void norm(std::vector<double> &A, size_t row, size_t column)
{
    size_t i, j;
    for (i = 0; i < row; i++)
    {
        auto max = A[i * column];
        for (j = 1; j < column; j++)
        {
            if (max < A[i * column + j])
            {
                max = A[i * column + j];
            }
        }
        for (j = 0; j < column; j++)
        {
            A[i * column + j] -= max;
        }
    }
}

void print_mat(const std::vector<double> &A, size_t row, size_t column)
{
    size_t i, j;
    bool flag1, flag2 = false;
    for (i = 0; i < row; i++)
    {
        flag1 = false;
        if (i < 5 || row - i < 5)
        {
            for (j = 0; j < column; j++)
            {
                if (j < 5 || column - j < 5)
                {
                    printf("%-14lf", A[i * column + j]);
                }
                else if (!flag1)
                {
                    printf("...   ");
                    flag1 = true;
                }
            }
            printf("\n");
        }
        else if (!flag2)
        {
            printf("...   \n");
            flag2 = true;
        }
    }
    std::cout << row << " x " << column << std::endl;
}

void print_all_mat(const std::vector<double> &A, size_t row, size_t column)
{
    size_t i, j;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < column; j++)
        {
            std::cout << A[i * column + j] << " ";
        }
        std::cout << "\n";
    }
}
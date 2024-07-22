#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#define W_ROWS 16
#define W_COLS 2
#define X_COLS 14
#define Y_COLS 11

class PE
{
public:
    int weight;
    int neuron;
    int psum;

    PE() : weight(0), neuron(0), psum(0) {}

    void calc()
    {
        psum += weight * neuron;
    }

    void shiftWeight(int new_weight)
    {
        weight = new_weight;
    }

    void shiftNeuron(int new_neuron)
    {
        neuron = new_neuron;
    }
};

class Systolic
{
public:
    PE S[W_ROWS][X_COLS];

public:
    // void Init()
    // {
    //     for (int i = 0; i < W_ROWS; i++)
    //         for (int j = 0; j < X_COLS; j++)
    //         {
    //             S[i][j].psum = 0;
    //             S[i][j].weight = 0;
    //             S[i][j].neuron = 0;
    //         }
    // }
    void calc()
    {
        for (int i = 0; i < W_ROWS; i++)
            for (int j = 0; j < X_COLS; j++)
                S[i][j].calc();
    }
    void shift(int a[W_ROWS], int b[X_COLS])
    {
        // 水平方向传播矩阵A,a[N]是本次要被读入的列(left->right)
        for (int i = 0; i < W_ROWS; i++)
            for (int j = X_COLS - 1; j > 0; j--)
            {
                S[i][j].shiftWeight(S[i][j - 1].weight);
            }
        for (int i = 0; i < W_ROWS; i++)
            S[i][0].shiftWeight(a[i]);
        // 竖直方向上传播矩阵B,b[N]是本次要被读入的行(up->bottom)
        for (int i = W_ROWS - 1; i > 0; i--)
            for (int j = 0; j < X_COLS; j++)
            {
                S[i][j].shiftNeuron(S[i - 1][j].neuron);
            }
        for (int j = 0; j < X_COLS; j++)
            S[0][j].shiftNeuron(b[j]);
    }
    void Display()
    {
        std::cout << "W:" << std::endl;
        for (int i = 0; i < W_ROWS; i++)
        {
            for (int j = 0; j < W_COLS; j++)
                std::cout << std::hex << std::setw(2) << S[i][j].weight << ",";
            std::cout << std::endl;
        }
        std::cout << "X:" << std::endl;
        for (int i = 0; i < W_ROWS; i++)
        {
            for (int j = 0; j < X_COLS; j++)
                std::cout << std::hex << std::setw(2) << S[i][j].neuron << ",";
            std::cout << std::endl;
        }
    }
};

void systolic_mm(int A[W_ROWS][W_COLS], int B[W_COLS][X_COLS], int C[W_ROWS][X_COLS])
{
    Systolic S;
    // S.Init();
    int a[W_ROWS];
    int b[X_COLS];
    int clock = 0;
    while (clock <= W_ROWS + W_COLS + X_COLS - 3)
    {
        // 产生a[N]
        for (int i = 0; i < W_ROWS; i++)
            a[i] = (clock >= i && clock < W_COLS + i) ? A[i][clock - i] : 0;
        // std::cout << "W_one:" << std::endl;
        // one_a(a);
        // 产生b[N]
        for (int j = 0; j < X_COLS; j++)
            b[j] = (clock >= j && clock < W_COLS + j) ? B[clock - j][j] : 0;
        // std::cout << "X_one:" << std::endl;
        // one_b(b);
        S.shift(a, b);
        S.calc();
        // std::cout << "clock=" << clock << std::endl;
        // S.Display();
        clock++;
    }
    for (int i = 0; i < W_ROWS; i++)
        for (int j = 0; j < X_COLS; j++)
            C[i][j] = S.S[i][j].psum;
    return;
}
void Matrix_Mult(int A[W_ROWS][W_COLS], int B[W_COLS][X_COLS], int C[W_ROWS][X_COLS])
{
    for (int i = 0; i < W_ROWS; i++)
        for (int j = 0; j < X_COLS; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < W_COLS; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return;
}

bool Compare(int O1[W_ROWS][X_COLS], int O2[W_ROWS][X_COLS])
{
    for (int i = 0; i < W_ROWS; i++)
        for (int j = 0; j < X_COLS; j++)
            if (O1[i][j] != O2[i][j])
                return false;
    return true;
}

void Print_W(int A[W_ROWS][W_COLS])
{
    for (int i = 0; i < W_ROWS; i++)
    {
        for (int j = 0; j < W_COLS; j++)
            std::cout << std::hex << std::setw(2) << A[i][j] << ",";
        std::cout << std::endl;
    }
}

void Print_X(int A[W_COLS][X_COLS])
{
    for (int i = 0; i < W_COLS; i++)
    {
        for (int j = 0; j < X_COLS; j++)
            std::cout << std::hex << std::setw(2) << A[i][j] << ",";
        std::cout << std::endl;
    }
}

void Print_Y(int A[W_ROWS][X_COLS])
{
    for (int i = 0; i < W_ROWS; i++)
    {
        for (int j = 0; j < X_COLS; j++)
            std::cout << std::hex << std::setw(2) << A[i][j] << ",";
        std::cout << std::endl;
    }
}

void one_a(int A[W_ROWS])
{
    for (int i = 0; i < W_ROWS; i++)
    {
        std::cout << std::hex << std::setw(2) << A[i] << ",";
    }
    std::cout << std::endl;
}

void one_b(int A[X_COLS])
{
    for (int i = 0; i < X_COLS; i++)
    {
        std::cout << std::hex << std::setw(2) << A[i] << ",";
    }
    std::cout << std::endl;
}

int main()
{
    srand(time(0));
    int C1[W_ROWS][X_COLS];
    int C2[W_ROWS][X_COLS];
    int W[W_ROWS][W_COLS];
    int X[W_COLS][X_COLS];
    int n;
    std::cout << "Input check round n:";
    std::cin >> n;
    std::cout << "W_ROWS=" << W_ROWS << std::endl;
    std::cout << "W_COLS=" << W_COLS << std::endl;
    std::cout << "X_COLS=" << X_COLS << std::endl;
    int i = 0;
    while (i++ < n)
    {
        for (int i = 0; i < W_ROWS; i++)
            for (int j = 0; j < W_COLS; j++)
            {
                W[i][j] = rand() % 10;
                // W[i][j] = rand() % 255;
            }
        for (int i = 0; i < W_COLS; i++)
            for (int j = 0; j < X_COLS; j++)
            {
                X[i][j] = rand() % 10;
                // X[i][j] = rand() % 255;
            }
        // std::cout << "W:" << std::endl;
        // Print_W(W);
        // std::cout << "X:" << std::endl;
        // Print_X(X);
        Matrix_Mult(W, X, C1);
        // std::cout << "groundtruth:" << std::endl;
        // Print_Y(C1);
        systolic_mm(W, X, C2);
        // std::cout << "my_answer:" << std::endl;
        // Print_Y(C2);
        bool is_right = Compare(C1, C2);
        if (!is_right)
        {
            std::cout << "error" << std::endl;
            break;
        }
        std::cout << "Compare groundtruth and my_answer,and the result is " << std::boolalpha << is_right << std::endl;
    }
    return 0;
}

#include "DLU.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"
#include "CU.hpp"
#include "VU.hpp"

#include <ctime>
#include <format>

size_t M = 10;
size_t K = 19;
size_t N = 5;
size_t M0 = 4;
size_t K0 = 6;
size_t N0 = 3;

#define CEIL(X, M) ((X + M - 1) / M)

std::vector<uint8_t> getCacheLine(std::vector<uint8_t> m_LMB, size_t M1, size_t K1, size_t M, size_t K, size_t M0, size_t K0)
{
    std::vector<uint8_t> cacheLine(M0 * K0);
    for (size_t i = 0; i < M0; i++)
    {
        for (size_t j = 0; j < K0; j++)
        {
            auto M_i = M1 * M0 + i;
            auto K_j = K1 * K0 + j;
            if (M_i < M && K_j < K)
                cacheLine[i * K0 + j] = m_LMB[M_i * K + K_j];
            else
            {
                cacheLine[i * K0 + j] = 0;
            }
        }
    }
    return cacheLine;
}

std::vector<uint8_t> getWeightCacheLine(std::vector<uint8_t> m_RMB, size_t N1, size_t K1, size_t N, size_t K, size_t N0, size_t K0)
{
    std::vector<uint8_t> cacheLine(N0 * K0);
    for (size_t i = 0; i < N0; i++)
    {
        for (size_t j = 0; j < K0; j++)
        {
            auto N_i = N1 * N0 + i;
            auto K_j = K1 * K0 + j;
            if (N_i < N && K_j < K)
                cacheLine[i * K0 + j] = m_RMB[N_i * K + K_j];
            else
            {
                cacheLine[i * K0 + j] = 0;
            }
        }
    }
    return cacheLine;
}

void display_F(std::vector<uint8_t> &f)
{
    std::cerr << std::format("M: {}, K: {}", M, K) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < M; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < K; ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * K + j));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}
void display_W(std::vector<uint8_t> &f)
{
    std::cerr << std::format("K: {}, N: {}", K, N) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < N; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < K; ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * K + j));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}
void print_F0(std::vector<uint8_t> &f)
{
    std::cerr << std::format("M0: {}, K0: {}", M0, K0) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K0; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K0; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < M0; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < K0; ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * K0 + j));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}
void print_W0(std::vector<uint8_t> &f)
{
    std::cerr << std::format("K0: {}, N0: {}", K0, N0) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K0; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < K0; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < N0; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < K0; ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * K0 + j));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}
void print_RES0(std::vector<uint8_t> &f)
{
    std::cerr << std::format("M0: {}, N0: {}", M0, N0) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < N0; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < N0; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < M0; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < N0; ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * N0 + j));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}
void print_PSB(std::vector<uint8_t> &f)
{
    std::cerr << std::format("M: {}, N: {}", M, N) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < N0 * CEIL(N, N0); ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < N0 * CEIL(N, N0); ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < M0 * CEIL(M, M0); ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < N0 * CEIL(N, N0); ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * N0 * CEIL(N, N0) + j));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}
std::vector<uint8_t> matmulCtrl(std::vector<uint8_t> &F, std::vector<uint8_t> &W, size_t M1, size_t K1, size_t N1,
                                size_t M, size_t K, size_t N,
                                size_t M0, size_t K0, size_t N0)
{
    std::vector<uint8_t> LMB_line(M0 * K0);
    std::vector<uint8_t> RMB_line(K0 * N0);
    LMB_line = getCacheLine(F, 0, 1, M, K, M0, K0);
    RMB_line = getCacheLine(W, 0, 2, N, K, N0, K0);
    // print_F0(LMB_line);
    // print_W0(RMB_line);
    std::vector<uint8_t> result(M0 * N0);
    for (size_t i = 0; i < M0; i++)
    {
        for (size_t j = 0; j < N0; j++)
        {
            for (size_t k = 0; k < K0; k++)
            {
                result[i * N0 + j] += LMB_line[i * K0 + k] * RMB_line[j * K0 + k];
            }
        }
    }
    // print_RES0(result);
    // PSB = (M1*M0) * (N1*N0)
    std::vector<uint8_t> m_PSB(M0 * CEIL(M, M0) * N0 * CEIL(N, N0));
    for (size_t i = 0; i < M0; i++)
    {
        for (size_t j = 0; j < N0; j++)
        {
            auto M_i = M1 * M0 + i;
            auto N_j = N1 * N0 + j;
            if (M_i < M && N_j < N)
                m_PSB[M_i * N + N_j] += result[i * N0 + j];
            else
            {
                m_PSB[M_i * N + N_j] += 0;
            }
        }
    }
    return m_PSB;
}
int main()
{
    std::vector<uint8_t> F(M * K);
    std::vector<uint8_t> W(M * K);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < K; ++j)
        {
            F.at(i * K + j) = rand() % 10;
        }
    display_F(F);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            W.at(i * N + j) = rand() % 10;
        }
    std::vector<uint8_t> PSB(M0 * CEIL(M, M0) * N0 * CEIL(N, N0));
    display_W(W);
    for (size_t i = 0; i < CEIL(M, M0); ++i)
    {
        // finished adding Kh*Kw dimensions
        for (size_t j = 0; j < CEIL(K, K0); ++j)
        {
            // Done computation along the Co0 dimension
            for (size_t k = 0; k < CEIL(N, N0); ++k)
            {
                PSB = matmulCtrl(F, W, i, j, k, M, K, N, M0, K0, N0);
            }
        }
    }
    print_PSB(PSB);
}
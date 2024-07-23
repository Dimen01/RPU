#ifndef RDTU_HPP
#define RDTU_HPP

#include "DLU.hpp"

class RDTU
{
public:
    RDTU(DLU &DLU, size_t size_cacheline_RMB, size_t num_cacheline_RMB, size_t Co)
        : L1_cache(DLU), m_RMB(size_cacheline_RMB * num_cacheline_RMB), m_MMB(Co) {}

    // load into RMB as a (Co0,Kh*Kw) matrix,
    // there is essentially no data layout transformation here
    void loadWeight(size_t Weight_L1_addr, size_t Co0, size_t Ci0, size_t Kh, size_t Kw)
    {
#define OFFSETRDTU(oi, ki, kj) (oi * Kh * Kw + ki * Kw + kj)
        for (size_t oi = 0; oi < Co0; ++oi)
        {
            for (size_t ki = 0; ki < Kh; ++ki)
            {
                for (size_t kj = 0; kj < Kw; ++kj)
                {
                    std::vector<uint8_t> L1Cacheline =
                        L1_cache.getWeightCacheLine(Weight_L1_addr, oi, ki, kj, Ci0, Kh, Kw);
                    for (size_t c = 0; c < Ci0; c++)
                    {
                        m_RMB[OFFSETRDTU(oi, ki, kj) * Ci0 + c] =
                            L1Cacheline[c];
                    }
                }
            }
        }
    }

    // Get the elements(CubeCacheLine) in row i and column j of
    // the matrix (Co0,Kh*Kw) in RMB
    std::vector<uint8_t> getWeightCacheLine(size_t N1, size_t K1, size_t N, size_t K, size_t N0, size_t K0)
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

private:
    DLU &L1_cache;
    std::vector<uint8_t> m_RMB;
    std::vector<uint8_t> m_MMB;
};

#endif // RDTU_HPP
#ifndef LDTU_HPP
#define LDTU_HPP

#include "DLU.hpp"

class LDTU
{
public:
    LDTU(DLU &DLU, size_t size_cacheline, size_t num_cacheline)
        : L1_cache(DLU), m_LMB(size_cacheline * num_cacheline){};

    // Matrix im2col to (H0*W0,Kh*Kw) in LMB from (H0+ kh-1,W0+ kw-1) subPaddedFeature in L1
    void load_im2col(size_t Feature_L1_addr, size_t H0, size_t W0, size_t C0,
                     size_t Kh, size_t Kw)
    {
#define OFFSETLDTU(i, j, ki, kj) ((i * W0 + j) * (Kh * Kw) + (ki * Kw + kj))
        // row
        for (size_t i = 0; i < H0; i++)
        {
            for (size_t j = 0; j < W0; j++)
            {
                // col
                for (size_t ki = 0; ki < Kh; ki++)
                {
                    for (size_t kj = 0; kj < Kw; kj++)
                    {
                        std::vector<uint8_t> L1Cacheline =
                            L1_cache.getFeatureCacheLine(Feature_L1_addr, i + ki, j + kj, W0, C0, Kw);
                        {
                            // channel
                            for (size_t c = 0; c < C0; c++)
                            {
                                m_LMB[OFFSETLDTU(i, j, ki, kj) * C0 + c] = L1Cacheline[c];
                            }
                        }
                    }
                }
            }
        }
    }

    // Get the subMatrix (M0,K0) in row i and column j of
    // the matrix (H0*W0,Kh*Kw*C0) in LMB
    std::vector<uint8_t> getCacheLine(size_t M1, size_t K1, size_t M, size_t K, size_t M0, size_t K0)
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

private:
    DLU &L1_cache;
    std::vector<uint8_t> m_LMB;
};

#endif // LDTU_HPP
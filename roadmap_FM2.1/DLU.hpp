#ifndef DLU_HPP
#define DLU_HPP

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <vector>

class DLU
{
public:
    DLU(std::vector<uint8_t> &L2, size_t size_cacheline, size_t num_cacheline)
        : L2_cache(L2), m_L1_cache(size_cacheline * num_cacheline) {}

    // Load subfeatures of size (H0+kh-1, W0+Kw-1) centered at subblocks (H1,W1,C1)
    // to addr (Kh/2,Kw/2 padding in all four directions, out of bounds is 0), stored in L1
    void loadSubPaddedFeature(size_t Feature_L1_addr, size_t H1, size_t W1, size_t C1,
                              size_t Feature_L2_addr, size_t H, size_t W, size_t C,
                              size_t H0, size_t W0, size_t C0, size_t Kh, size_t Kw)
    {
        // Reminder: A subFeature in L1 is a (H0+Kh-1,W0+Kw-1) matrix,
        // where each element is a cacheline with C0 channels.
        // Iterate over all elements to load i <- [0, H0+Kh-1), j <- [0, W0+Kw-1)
        for (size_t i = 0; i < H0 + Kh - 1; i++)
        {
            for (size_t j = 0; j < W0 + Kw - 1; j++)
            {
// Calculate the offset of the current element's addr relative to L1
// row major storage is used
#define OFFSETDLUFeature(i, j) (i * (W0 + Kw - 1) + j)
                // Read elements from L2, temporarily load them one by one
                for (size_t c = 0; c < C0; c++)
                {
                    // Calculate the address of the current element in L2
                    size_t L2_addr_row = H1 * H0 + (i - Kh / 2);
                    size_t L2_addr_col = W1 * W0 + (j - Kw / 2);
                    size_t L2_addr_cha = C1 * C0 + c;
                    // Read elements from L2
                    // judge whether (im_row, im_col, im_cha) is out of bound
                    if (L2_addr_row < H && L2_addr_col < W && L2_addr_cha < C)
                    {
                        size_t L2_addr_offset =
                            L2_addr_row * W * C +
                            L2_addr_col * C +
                            L2_addr_cha;
                        m_L1_cache[Feature_L1_addr + OFFSETDLUFeature(i, j) * C0 + c] =
                            L2_cache.at(Feature_L2_addr + L2_addr_offset);
                    }
                    else
                    {
                        m_L1_cache[Feature_L1_addr + OFFSETDLUFeature(i, j) * C0 + c] = 0;
                    }
                }
            }
        }
    }

    // Load subweight of size (Co0,Kh,Kw) centered at subblocks (Co1,Ci1) into addr and store in L1
    void loadSubWeight(size_t Weight_L1_addr, size_t Co1, size_t Ci1,
                       size_t Weight_L2_addr, size_t Co, size_t Ci,
                       size_t Co0, size_t Ci0, size_t Kh, size_t Kw)
    {
        // Reminder:subWeight is a (Co0,Kh,Kw) matrix in L1,
        // where each element is a cacheline, containing Ci0 channels.
        // Iterate over all the elements to load
        // Co0 Co channels: oi <- [0, Co0)
        for (size_t oi = 0; oi < Co0; oi++)
        {
            // Iterate over each element of weight ki <- [0, Kh), kj <- [0, Kw)
            for (size_t ki = 0; ki < Kh; ki++)
            {
                for (size_t kj = 0; kj < Kw; kj++)
                {
// Calculate the offset of the current element's addr relative to L1
// row major storage is used
#define OFFSETDLUWeight(oi, ki, kj) (oi * Kh * Kw + ki * Kw + kj)
                    // Load elements from L2, temporarily Ci0 elements one by one
                    for (size_t ci = 0; ci < Ci0; ci++)
                    {
                        // Calculate the address of the current element in L2
                        size_t L2_addr_cho = Co1 * Co0 + oi;
                        size_t L2_addr_row = ki;
                        size_t L2_addr_col = kj;
                        size_t L2_addr_chi = Ci1 * Ci0 + ci;
                        // Read elements from L2
                        // judge whether (im_cho, im_chi) is out of bound
                        if (L2_addr_cho < Co && L2_addr_chi < Ci)
                        {
                            size_t L2_addr_offset =
                                L2_addr_cho * Kh * Kw * Ci +
                                L2_addr_row * Kw * Ci +
                                L2_addr_col * Ci +
                                L2_addr_chi;
                            m_L1_cache[Weight_L1_addr + OFFSETDLUWeight(oi, ki, kj) * Ci0 + ci] =
                                L2_cache.at(Weight_L2_addr + L2_addr_offset);
                        }
                        else
                        {
                            m_L1_cache[Weight_L1_addr + OFFSETDLUWeight(oi, ki, kj) * Ci0 + ci] = 0;
                        }
                    }
                }
            }
        }
    }

    // interpret the memory from addr as a (H0+Kh-1,W0+Kw-1) Feature matrix
    // with each entry being a cacheline(C0 elements)
    std::vector<uint8_t> getFeatureCacheLine(size_t Feature_L1_addr, size_t i, size_t j,
                                             size_t W0, size_t C0, size_t Kw)
    {
        std::vector<uint8_t> cacheLine(C0);
        for (size_t c = 0; c < C0; c++)
        {
            cacheLine[c] = m_L1_cache[Feature_L1_addr + (i * (W0 + Kw - 1) + j) * C0 + c];
        }
        return cacheLine;
    }

    // interpret the memory from addr as a (H0+Kh-1,W0+Kw-1) Feature matrix
    // with each entry being a cacheline(Ci0 elements)
    std::vector<uint8_t> getWeightCacheLine(size_t Weight_L1_addr, size_t oi, size_t ki, size_t kj,
                                            size_t Ci0, size_t Kh, size_t Kw)
    {
        std::vector<uint8_t> cacheLine(Ci0);
        for (size_t c = 0; c < Ci0; c++)
        {
            cacheLine[c] = m_L1_cache[Weight_L1_addr + (oi * Kh * Kw + ki * Kw + kj) * Ci0 + c];
        }
        return cacheLine;
    }

private:
    std::vector<uint8_t> &L2_cache;
    std::vector<uint8_t> m_L1_cache;
};

#endif // DLU_HPP
/*
 * @Author: Dimen01 miaohw2022@163.com
 * @Date: 2024-07-23 20:44:40
 * @LastEditors: Dimen01 miaohw2022@163.com
 * @LastEditTime: 2024-07-23 21:39:19
 * @FilePath: \RPU\roadmap_FM2.1\CU.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef CU_HPP
#define CU_HPP

#include "LDTU.hpp"
#include "RDTU.hpp"

#include <iostream>

class CU
{
public:
    CU(LDTU &LDTU, RDTU &RDTU, size_t size_cacheline, size_t num_cacheline)
        : LMB(LDTU), RMB_MMB(RDTU), m_PSB(size_cacheline * num_cacheline) {}

    void clearPSB() { m_PSB.clear(); }

    std::vector<uint8_t> &getPSB() { return m_PSB; }

    std::vector<uint8_t> const getPSB() const { return m_PSB; }

    void matmulCtrl(size_t M1, size_t K1, size_t N1,
                    size_t M, size_t K, size_t N,
                    size_t M0, size_t K0, size_t N0)
    {
        // M : H0 * W0
        // K : Kh * Kw
        // N : Co0

        // M1 is the index with [0, CEIL(H0 * W0, M0))
        // K1 is the index with [0, Ceil(Kh * Kw, K0)
        // N1 is the index with [0, CEIL(Co0, N0))

        // LMB : (H0 * W0) * (Kh * Kw) = M * K = (M1 * M0) * (K1 * K0)
        // RMB : (Co0) * (Kh * Kw) = N * K = (N1 * N0) * (K1 * K0)
        // PSB : (H0 * W0) * N This is still HWC,
        // but note that N0 = C0 = 16,
        // the last dimension has 16 channels to fill up a cacheline,
        // if Co0 < C0 (N < N0) then there will be padding

        // LDTU
        std::vector<uint8_t> LMB_line = LMB.getCacheLine(M1, K1, M, K, M0, K0);

        // RDTU
        std::vector<uint8_t> RMB_line = RMB_MMB.getWeightCacheLine(N1, K1, N, K, N0, K0);

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

        // PSB = (M1*M0) * (N1*N0)
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
    }

private:
    LDTU &LMB;
    RDTU &RMB_MMB;
    std::vector<uint8_t> m_PSB;
};

#endif // CU_HPP
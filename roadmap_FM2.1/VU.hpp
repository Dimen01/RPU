#ifndef VU_HPP
#define VU_HPP

#include "DLU.hpp"
#include "CU.hpp"

class VU
{
public:
    VU(DLU &DLU, CU &CU) : DLU(DLU), CU(CU) {}

    // PSB = (M1*M0) * (N1*N0) = (H0 * W0) * Co0
    // res = (H1*H0) * (W1*W0) * (Co1*Co0)
    void LoadOut(std::vector<uint8_t> &res,
                 size_t H1, size_t W1, size_t Co1,
                 size_t H0, size_t W0, size_t Co0,
                 size_t H, size_t W, size_t Co,
                 size_t Kh, size_t Kw, size_t stride)
    {
        size_t retH = (H - Kh + Kh / 2 * 2) / stride + 1;
        size_t retW = (W - Kw + Kw / 2 * 2) / stride + 1;
        for (size_t i = 0; i < H0; i++)
        {
            for (size_t j = 0; j < W0; j++)
            {
                for (size_t k = 0; k < Co0; k++)
                {
                    auto H_i = H1 * H0 + i;
                    auto W_j = W1 * W0 + j;
                    auto C_k = Co1 * Co0 + k;
                    if (H_i < retH && W_j < retW && C_k < Co)
                        res[H_i * retW * Co + W_j * Co + C_k] =
                            CU.getPSB()[i * W0 * Co0 + j * Co0 + k];
                }
            }
        }
    }

private:
    DLU &DLU;
    CU &CU;
};

#endif // VU_HPP
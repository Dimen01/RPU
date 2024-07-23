#include "DLU.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"
#include "CU.hpp"
#include "VU.hpp"

#include <ctime>
#include <format>

std::vector<uint8_t> l2_i(1024 * 1024);
DLU dlu_i(l2_i, 16, 64 * 1024);
LDTU ldtu_i(dlu_i, 16, 4 * 1024);
RDTU rdtu_i(dlu_i, 16, 4 * 1024, 32 * 1024);
CU cu_i(ldtu_i, rdtu_i, 16, 8 * 1024);
VU vu_i(dlu_i, cu_i);

// How to partion the feature and weight
size_t Co0 = 5;
size_t H0 = 8;
size_t W0 = 3;
size_t C0 = 16;
size_t Ci0 = C0;
size_t stride = 1;

// ----- Feature in L2 -----
size_t H = 22;
size_t W = 19;
size_t C = 18;
size_t Feature_L2_addr = 0;
size_t Feature_L1_addr = 0;

// ----- Weight in L2 -----
size_t Co = 18;
size_t Kh = 3;
size_t Kw = Kh;
size_t Ci = C;
size_t Weight_L2_addr = H * W * C;
size_t Weight_L1_addr = H0 * W0 * Kh * Kw * C0;

// define M, N, K
size_t M = H0 * W0;
size_t K = Kh * Kw * Ci0;
size_t N = Co0;
size_t M0 = 16;
size_t K0 = 16;
size_t N0 = 16;

#define CEIL(X, M) ((X + M - 1) / M)

#define FEATURE_INDEX(H, W, C, i, j, k) (i * W * C + j * C + k)
#define WEIGHT_INDEX(Co, Kh, Kw, Ci, oi, ki, kj, ii) \
    (oi * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ii)

// ----- computation -----
size_t Co1 = CEIL(Co, Co0);
size_t H1 = CEIL(H, H0);
size_t W1 = CEIL(W, W0);
size_t Ci1 = CEIL(Ci, C0);

std::vector<uint8_t> golden_WconvF(std::vector<uint8_t> &f, std::vector<uint8_t> &w)
{
    // kernel should be a square matrix whose sidelength should be an odd number
    size_t retH = (H - Kh + Kh / 2 * 2) / stride + 1;
    size_t retW = (W - Kw + Kw / 2 * 2) / stride + 1;
    std::vector<uint8_t> res(retH * retW * Co);
    auto row_pad = Kh / 2;
    auto col_pad = Kw / 2;
    for (size_t co = 0; co < Co; ++co)
    {
        for (size_t i = 0; i < retH; ++i)
        {
            for (size_t j = 0; j < retW; ++j)
            {
                for (size_t ci = 0; ci < Ci; ++ci)
                {
                    for (size_t ki = 0; ki < Kh; ++ki)
                    {
                        for (size_t kj = 0; kj < Kw; ++kj)
                        {
                            size_t im_row = stride * i - row_pad + ki;
                            size_t im_col = stride * j - col_pad + kj;
                            if (im_row < H && im_col < W && ci < C)
                            {
                                res.at(i * retW * Co + j * Co + co) += f.at(im_row * W * C + im_col * C + ci) * w.at(co * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ci);
                            }
                            else
                            {
                                res.at(i * retW * Co + j * Co + co) += 0;
                            }
                        }
                    }
                }
            }
        }
    }
    return res;
}

void display(std::vector<uint8_t> &f, size_t c)
{
    assert(c < C);
    std::cerr << std::format("H: {}, W: {}", H, W) << "\n";
    std::cerr << std::format("Channel: {}", c) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < W; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < W; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < H; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < W; ++j)
        {
            std::cerr << std::format("{:5}", (int)f.at(i * W * C + j * C + c));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}

void checkDisplay(std::vector<uint8_t> &golden_res, std::vector<uint8_t> &res)
{
    for (size_t k = 0; k < Co; ++k)
    {
        std::cerr << "reference: " << std::endl;
        display(golden_res, k);
        std::cerr << "result: " << std::endl;
        display(res, k);
    }
    assert(res == golden_res);
}

int main()
{
    srand(time(0));

    // ----- Feature in L2 -----
    std::vector<uint8_t> feature(H * W * C);
    for (size_t i = 0; i < H; ++i)
        for (size_t j = 0; j < W; ++j)
            for (size_t k = 0; k < C; ++k)
            {
                auto val = j + 10 * k;
                l2_i.at(Feature_L2_addr + FEATURE_INDEX(H, W, C, i, j, k)) = val;
                feature.at(i * W * C + j * C + k) = val;
            }

    // ----- Weight in L2 -----
    std::vector<uint8_t> weight(Co * Kh * Kw * Ci);
    for (size_t oi = 0; oi < Co; ++oi)
        for (size_t ki = 0; ki < Kh; ++ki)
            for (size_t kj = 0; kj < Kw; ++kj)
                for (size_t ii = 0; ii < Ci; ++ii)
                {
                    auto val = 1;
                    l2_i.at(Weight_L2_addr + WEIGHT_INDEX(Co, Kh, Kw, Ci, oi, ki, kj, ii)) = val;
                    weight.at(oi * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ii) = val;
                }

    // For easy comparison, use Feature to store PSB transferred results
    std::vector<uint8_t> res(H * W * Co);

    for (size_t oi = 0; oi < Co1; ++oi)
        for (size_t i = 0; i < H1; ++i)
            for (size_t j = 0; j < W1; ++j)
            {
                // clear PSB,ii loop is finished and PSB is the result of a (H0,W0,Co0) sub-block
                // if Co0 < C0 (N < N0) then there will be padding
                cu_i.clearPSB();
                // Continue to do Ci1 times to complete the accumulation of Ci dimensions
                for (size_t ii = 0; ii < Ci1; ++ii)
                {
                    dlu_i.loadSubPaddedFeature(Feature_L1_addr, i, j, ii, Feature_L2_addr, H, W, C, H0, W0, C0, Kh, Kw);
                    dlu_i.loadSubWeight(Weight_L1_addr, oi, ii, Weight_L2_addr, Co, Ci, Co0, Ci0, Kh, Kw);
                    ldtu_i.load_im2col(Feature_L1_addr, H0, W0, C0, Kh, Kw);
                    rdtu_i.loadWeight(Weight_L1_addr, Co0, Ci0, Kh, Kw);

                    // CU calculation
                    // Done computation along the H0*W0 dimension
                    for (size_t i = 0; i < CEIL(M, M0); ++i)
                        // finished adding Kh*Kw dimensions
                        for (size_t j = 0; j < CEIL(K, K0); ++j)
                            // Done computation along the Co0 dimension
                            for (size_t k = 0; k < CEIL(N, N0); ++k)
                                cu_i.matmulCtrl(i, j, k, M, K, N, M0, K0, N0);
                }

                vu_i.LoadOut(res, i, j, oi, H0, W0, Co0, H, W, Co, Kh, Kw, stride);
            }

    auto ref_result = golden_WconvF(feature, weight);
    checkDisplay(ref_result, res);
    puts("Passed");
}
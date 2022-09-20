/*
 * This file is part of FFmpeg.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cuda/vector_helpers.cuh"

// FFmpeg passes pitch in bytes, CUDA uses potentially larger types
#define FIXED_PITCH \
    (dst_pitch/sizeof(*dst_0))

template<typename T>
__device__ static inline T Subsample_Bilinear(cudaTextureObject_t tex,
                                              int xo, int yo,
                                              int dst_width, int dst_height,
                                              int src_width, int src_height,
                                              int bit_depth, float param)
{
    float hscale = (float)src_width / (float)dst_width;
    float vscale = (float)src_height / (float)dst_height;
    float xi = (xo + 0.5f) * hscale;
    float yi = (yo + 0.5f) * vscale;
    // 3-tap filter weights are {wh,1.0,wh} and {wv,1.0,wv}
    float wh = min(max(0.5f * (hscale - 1.0f), 0.0f), 1.0f);
    float wv = min(max(0.5f * (vscale - 1.0f), 0.0f), 1.0f);
    // Convert weights to two bilinear weights -> {wh,1.0,wh} -> {wh,0.5,0} + {0,0.5,wh}
    float dx = wh / (0.5f + wh);
    float dy = wv / (0.5f + wv);

    intT r;
    vec_set_scalar(r, 2);
    r += tex2D<T>(tex, xi - dx, yi - dy);
    r += tex2D<T>(tex, xi + dx, yi - dy);
    r += tex2D<T>(tex, xi - dx, yi + dy);
    r += tex2D<T>(tex, xi + dx, yi + dy);

    T res;
    vec_set(res, r >> 2);

    return res;
}

extern "C" {

__constant__ uchar font[11][8][8] = {
    { // 0
        0,0,0,0,0,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,0,1,0,0,
        0,0,1,0,0,1,0,0,
        0,0,1,0,0,1,0,0,
        0,0,1,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 1
        0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,1,1,1,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 2
        0,0,0,0,0,0,0,0,
        0,0,1,1,1,0,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,0,0,0,0,
        0,0,1,0,0,0,0,0,
        0,0,0,1,1,1,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 3
        0,0,0,0,0,0,0,0,
        0,0,1,1,1,0,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,0,0,1,0,0,
        0,0,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 4
        0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,1,0,0,0,
        0,1,1,1,1,1,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 5
        0,0,0,0,0,0,0,0,
        0,0,0,1,1,1,0,0,
        0,0,1,0,0,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,0,0,1,0,0,
        0,0,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 6
        0,0,0,0,0,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,0,0,0,0,
        0,0,1,1,1,0,0,0,
        0,0,1,0,0,1,0,0,
        0,0,1,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 7
        0,0,0,0,0,0,0,0,
        0,0,1,1,1,1,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 8
        0,0,0,0,0,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,0,1,0,0,
        0,0,1,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // 9
        0,0,0,0,0,0,0,0,
        0,0,0,1,1,0,0,0,
        0,0,1,0,0,1,0,0,
        0,0,1,0,0,1,0,0,
        0,0,0,1,1,1,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,0,0,0,0,0,
    },
    { // Slash
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,0,1,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,
    },
};

__global__ void Subsample_Bilinear_nv12_nv12(
        cudaTextureObject_t src_tex_0, cudaTextureObject_t src_tex_1,
        cudaTextureObject_t src_tex_2, cudaTextureObject_t src_tex_3,
        uchar *dst_0, uchar *dst_1, uchar *dst_2, uchar *dst_3,
        int dst_width, int dst_height, int dst_pitch,
        int src_width, int src_height, float param,
        char *frame_rate_num, int frame_rate_num_digits,
        char *frame_rate_den, int frame_rate_den_digits)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (yo >= dst_height || xo >= dst_width) return;

    uchar y = Subsample_Bilinear<uchar>(src_tex_0, xo, yo, dst_width, dst_height, src_width, src_height, 8, 0.0f);

    // Draw FPS
    int font_scale_log2 = 2;
    int frame_rate_msg_size = frame_rate_num_digits + 1 + frame_rate_den_digits;
    int font_size = 8 << font_scale_log2;

    if (xo < frame_rate_msg_size * font_size && yo < font_size)
    {
        int symbol_pos = xo / font_size;
        char symbol = 10; // Slash
        if (symbol_pos < frame_rate_num_digits)
        {
            symbol = frame_rate_num[symbol_pos] - '0';
        }
        else if (symbol_pos > frame_rate_num_digits)
        {
            symbol = frame_rate_den[symbol_pos - 1 - frame_rate_num_digits] - '0';
        }

        int fontx = (xo>>font_scale_log2) & 0x7;
        int fonty = (yo>>font_scale_log2) & 0x7;

        y = font[symbol][fonty][fontx] * 255;
    }

    dst_0[yo*FIXED_PITCH+xo] = y;
}

__global__ void Subsample_Bilinear_nv12_nv12_uv(
        cudaTextureObject_t src_tex_0, cudaTextureObject_t src_tex_1,
        cudaTextureObject_t src_tex_2, cudaTextureObject_t src_tex_3,
        uchar2 *dst_0, uchar2 *dst_1, uchar2 *dst_2, uchar2 *dst_3,
        int dst_width, int dst_height, int dst_pitch,
        int src_width, int src_height, float param,
        char *frame_rate_num, int frame_rate_num_digits,
        char *frame_rate_den, int frame_rate_den_digits)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (yo >= dst_height || xo >= dst_width) return;

    uchar2 uv = Subsample_Bilinear<uchar2>(src_tex_1, xo, yo, dst_width, dst_height, src_width, src_height, 8, 0.0f);

    dst_1[yo*FIXED_PITCH+xo].x = uv.x;
    dst_1[yo*FIXED_PITCH+xo].y = uv.y;
}
}

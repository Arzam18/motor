#ifndef MOTOR_NNUE_HPP
#define MOTOR_NNUE_HPP

#include <algorithm>
#include <array>
#include <cstdint>

#include "incbin.hpp"

// Conditional SIMD Headers
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

constexpr unsigned int HIDDEN_SIZE = 1536;
constexpr int QA = 403;
constexpr int QB = 81;
constexpr int Cells = 8;

constexpr std::array<int, 64> buckets = {
        0, 1, 2, 3, 11, 10, 9, 8,
        4, 4, 5, 5, 13, 13, 12, 12,
        6, 6, 6, 6, 14, 14, 14, 14,
        6, 6, 6, 6, 14, 14, 14, 14,
        7, 7, 7, 7, 15, 15, 15, 15,
        7, 7, 7, 7, 15, 15, 15, 15,
        7, 7, 7, 7, 15, 15, 15, 15,
        7, 7, 7, 7, 15, 15, 15, 15,
};

struct Weights {
    std::array<std::array<std::array<std::array<std::array<std::int16_t, HIDDEN_SIZE>, 64>, 6>, 2>, Cells> feature_weight;
    std::array<std::int16_t, HIDDEN_SIZE> feature_bias;
    std::array<std::int16_t, HIDDEN_SIZE> output_weight_STM;
    std::array<std::int16_t, HIDDEN_SIZE> output_weight_NSTM;
    std::int16_t output_bias;
};

INCBIN(Weights, "nnue.bin");
const Weights& weights = *reinterpret_cast<const Weights*>(gWeightsData);

enum class Operation {
    Set, Unset
};

inline int screlu(int x) {
    std::int16_t clamped = std::clamp(x, 0, QA);
    return clamped * clamped;
}

// SIMD Helper for Accumulator Addition / Subtraction
template <Operation op>
inline void vec_update(std::int16_t* acc, const std::int16_t* weights_ptr, std::size_t size) {
#if defined(__AVX2__)
    constexpr std::size_t step = 16;
    for (std::size_t i = 0; i < size; i += step) {
        auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        auto w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights_ptr + i));
        auto res = (op == Operation::Set) ? _mm256_add_epi16(a, w) : _mm256_sub_epi16(a, w);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), res);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    constexpr std::size_t step = 8;
    for (std::size_t i = 0; i < size; i += step) {
        int16x8_t a = vld1q_s16(acc + i);
        int16x8_t w = vld1q_s16(weights_ptr + i);
        int16x8_t res = (op == Operation::Set) ? vaddq_s16(a, w) : vsubq_s16(a, w);
        vst1q_s16(acc + i, res);
    }
#elif defined(__SSE2__)
    constexpr std::size_t step = 8;
    for (std::size_t i = 0; i < size; i += step) {
        auto a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(acc + i));
        auto w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights_ptr + i));
        auto res = (op == Operation::Set) ? _mm_add_epi16(a, w) : _mm_sub_epi16(a, w);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + i), res);
    }
#else
    if constexpr (op == Operation::Set) {
        for (std::size_t i = 0; i < size; i++) {
            acc[i] += weights_ptr[i];
        }
    } else {
        for (std::size_t i = 0; i < size; i++) {
            acc[i] -= weights_ptr[i];
        }
    }
#endif
}

template<std::uint16_t hidden_size>
struct accumulator_cache {
    std::array<std::array<std::int16_t, hidden_size>, 128> white_accumulator_stack;
    std::array<std::array<std::int16_t, hidden_size>, 128> black_accumulator_stack;
};

template<std::uint16_t hidden_size>
class perspective_network
{
public:
    std::array<std::array<std::int16_t, hidden_size>, 128> white_accumulator_stack;
    std::array<std::array<std::int16_t, hidden_size>, 128> black_accumulator_stack;
    unsigned int index;

    perspective_network() {
        refresh();
    }

    template <Color perspective>
    void refresh_current_accumulator() {
        if constexpr (perspective == White) {
            white_accumulator_stack[index] = weights.feature_bias;
        } else {
            black_accumulator_stack[index] = weights.feature_bias;
        }
    }

    void refresh() {
        white_accumulator_stack[0] = black_accumulator_stack[0] = weights.feature_bias;
        index = 0;
    }

    void push() {
        white_accumulator_stack[index + 1] = white_accumulator_stack[index];
        black_accumulator_stack[index + 1] = black_accumulator_stack[index];
        index++;
    }

    void pull() {
        index--;
    }

    inline int get_square_index(int square, int king_square) {
        return (king_square % 8 > 3) ? square ^ 7 : square;
    }

    template<Operation operation, Color perspective>
    void update_accumulator(const Piece piece, const Color color, const Square square, int king) {
        if constexpr (perspective == White) {
            const auto& white_weights = weights.feature_weight[buckets[king] % Cells][color][piece][get_square_index(square, king)];
            auto& white_accumulator = white_accumulator_stack[index];
            vec_update<operation>(white_accumulator.data(), white_weights.data(), hidden_size);
        } else {
            const auto& black_weights = weights.feature_weight[buckets[king ^ 56] % Cells][color ^ 1][piece][get_square_index(square, king) ^ 56];
            auto& black_accumulator = black_accumulator_stack[index];
            vec_update<operation>(black_accumulator.data(), black_weights.data(), hidden_size);
        }
    }

    template<Operation operation>
    void update_accumulator(const Piece piece, const Color color, const Square square, int wking, int bking) {
        const auto& white_weights = weights.feature_weight[buckets[wking] % Cells][color][piece][get_square_index(square, wking)];
        const auto& black_weights = weights.feature_weight[buckets[bking ^ 56] % Cells][color ^ 1][piece][get_square_index(square, bking) ^ 56];

        auto& white_accumulator = white_accumulator_stack[index];
        auto& black_accumulator = black_accumulator_stack[index];

        vec_update<operation>(white_accumulator.data(), white_weights.data(), hidden_size);
        vec_update<operation>(black_accumulator.data(), black_weights.data(), hidden_size);
    }

    template <Color color>
    std::int32_t evaluate() {
        const auto& stm_accumulator = color == White ? white_accumulator_stack[index] : black_accumulator_stack[index];
        const auto& nstm_accumulator = color == White ? black_accumulator_stack[index] : white_accumulator_stack[index];

#if defined(__AVX2__) || defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__SSE2__)
        std::int32_t sum = 0;
        sum += flatten(stm_accumulator.data(), weights.output_weight_STM.data());
        sum += flatten(nstm_accumulator.data(), weights.output_weight_NSTM.data());
#else
        std::int32_t sum = 0;
        for (std::size_t j = 0; j < hidden_size; j++) {
            sum += screlu(stm_accumulator[j]) * weights.output_weight_STM[j];
            sum += screlu(nstm_accumulator[j]) * weights.output_weight_NSTM[j];
        }
#endif
        return (sum / QA + weights.output_bias) * 400 / (QB * QA);
    }

private:
#if defined(__AVX2__)
    std::int32_t flatten(const std::int16_t* accumulator, const std::int16_t* weights_ptr) {
        constexpr int CHUNK = 16;
        auto sum = _mm256_setzero_si256();
        auto min = _mm256_setzero_si256();
        auto max = _mm256_set1_epi16(QA);
        for (int i = 0; i < hidden_size / CHUNK; i++) {
            auto us_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator + i * CHUNK));
            auto weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights_ptr + i * CHUNK));
            auto clamped = _mm256_min_epi16(_mm256_max_epi16(us_vector, min), max);
            auto mul = _mm256_madd_epi16(clamped, _mm256_mullo_epi16(clamped, weights_vec));
            sum = _mm256_add_epi32(sum, mul);
        }
        return horizontal_sum(sum);
    }

    std::int32_t horizontal_sum(const __m256i input_sum) {
        __m256i horizontal_sum_256 = _mm256_hadd_epi32(input_sum, input_sum);
        __m128i upper_128 = _mm256_extracti128_si256(horizontal_sum_256, 1);
        __m128i combined_128 = _mm_add_epi32(upper_128, _mm256_castsi256_si128(horizontal_sum_256));
        __m128i horizontal_sum_128 = _mm_hadd_epi32(combined_128, combined_128);
        return _mm_cvtsi128_si32(horizontal_sum_128);
    }

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::int32_t flatten(const std::int16_t* accumulator, const std::int16_t* weights_ptr) {
        int32x4_t sum_vec0 = vdupq_n_s32(0);
        int32x4_t sum_vec1 = vdupq_n_s32(0);
        const int16x8_t min_vec = vdupq_n_s16(0);
        const int16x8_t max_vec = vdupq_n_s16(QA);

        for (std::size_t i = 0; i < hidden_size; i += 8) {
            int16x8_t acc = vld1q_s16(accumulator + i);
            int16x8_t w   = vld1q_s16(weights_ptr + i);

            int16x8_t clamped = vminq_s16(vmaxq_s16(acc, min_vec), max_vec);
            int16x8_t cw = vmulq_s16(clamped, w);

            int32x4_t prod_low  = vmull_s16(vget_low_s16(clamped), vget_low_s16(cw));
            int32x4_t prod_high = vmull_s16(vget_high_s16(clamped), vget_high_s16(cw));

            sum_vec0 = vaddq_s32(sum_vec0, prod_low);
            sum_vec1 = vaddq_s32(sum_vec1, prod_high);
        }

        int32x4_t sum_vec = vaddq_s32(sum_vec0, sum_vec1);
#if defined(__aarch64__) || defined(_M_ARM64)
        return vaddvq_s32(sum_vec);
#else
        int32x2_t sum_halves = vadd_s32(vget_low_s32(sum_vec), vget_high_s32(sum_vec));
        return vget_lane_s32(vpadd_s32(sum_halves, sum_halves), 0);
#endif
    }

#elif defined(__SSE2__)
    std::int32_t flatten(const std::int16_t* accumulator, const std::int16_t* weights_ptr) {
        constexpr int CHUNK = 8;
        auto sum = _mm_setzero_si128();
        auto min = _mm_setzero_si128();
        auto max = _mm_set1_epi16(QA);
        for (int i = 0; i < hidden_size / CHUNK; i++) {
            auto us_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(accumulator + i * CHUNK));
            auto weights_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights_ptr + i * CHUNK));
            auto clamped = _mm_min_epi16(_mm_max_epi16(us_vector, min), max);
            auto mul = _mm_madd_epi16(clamped, _mm_mullo_epi16(clamped, weights_vec));
            sum = _mm_add_epi32(sum, mul);
        }
        __m128i hi64 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
        sum = _mm_add_epi32(sum, hi64);
        __m128i hi32 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1));
        sum = _mm_add_epi32(sum, hi32);
        return _mm_cvtsi128_si32(sum);
    }
#endif
};

extern perspective_network<HIDDEN_SIZE> network;

#endif // MOTOR_NNUE_HPP

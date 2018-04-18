#ifndef PSRDADA_CPP_MEERKAT_COMMON_HPP
#define PSRDADA_CPP_MEERKAT_COMMON_HPP

struct ComplexInt8
{
  int8_t x;
  int8_t y;
};

typedef ComplexInt8 weight_t;
typedef ComplexInt8 voltage_t;
typedef float power_t;
typedef float frequency_t;

#endif //PSRDADA_CPP_MEERKAT_COMMON_HPP
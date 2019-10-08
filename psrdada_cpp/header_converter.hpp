#ifndef PSRDADA_CPP_HEADER_CONVERTER_HPP
#define PSRDADA_CPP_HEADER_CONVERTER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <functional>

namespace psrdada_cpp {


template <class HandlerType>
class HeaderConverter
{
public:
    typedef std::function<void(RawBytes&, RawBytes&)> HeaderParserType;

public:
    HeaderConverter(HeaderParserType parser, HandlerType& handler);
    HeaderConverter(HeaderConverter const&) = delete;
    ~HeaderConverter();

    void init(RawBytes& block);

    bool operator()(RawBytes& block);

private:
    HeaderParserType _parser;
    HandlerType& _handler;
    char* _optr;
};


} //psrdada_cpp

#include "psrdada_cpp/detail/header_converter.cpp"
#endif //PSRDADA_CPP_HEADER_CONVERTER_HPP




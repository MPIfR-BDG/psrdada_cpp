#include "psrdada_cpp/header_converter.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <thread>

namespace psrdada_cpp {

    template <class HandlerType>
    HeaderConverter<HandlerType>::HeaderConverter(HeaderParserType parser, HandlerType& handler)
    : _parser(parser)
    , _handler(handler)
    {
        _optr = new char[4096];
    }

    template <class HandlerType>
    HeaderConverter<HandlerType>::~HeaderConverter()
    {
        delete[] _optr;
    }

    template <class HandlerType>
    void HeaderConverter<HandlerType>::init(RawBytes& block)
    {
        RawBytes outblock(_optr, block.total_bytes(), 0, false);
        _parser(block, outblock);
        outblock.used_bytes(outblock.total_bytes());
        _handler.init(outblock);
    }

    template <class HandlerType>
    bool HeaderConverter<HandlerType>::operator()(RawBytes& block)
    {
        _handler(block);
        return false;
    }

} //psrdada_cpp

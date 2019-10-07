#ifndef PSRDADA_CPP_NULLABLE_HANDLER_HPP
#define PSRDADA_CPP_NULLABLE_HANDLER_HPP

#include "psrdada_cpp/handler_switch.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"

namespace psrdada_cpp {

template <typename Handler>
class NullableHandler: public HandlerSwitch<Handler, NullSink>
{
public:
    explicit NullableHandler(Handler& handler)
    : HandlerSwitch(handler, _null){};
    NullableHandler(NullableHandler const&) = delete;
    ~NullableHandler();
    bool is_nulled() const;
    void unnull();
    void null();

private:
    NullSink _null;
};

} //psrdada_cpp

#include "psrdada_cpp/detail/nullable_handler.cpp"

#endif //PSRDADA_CPP_NULLABLE_HANDLER_HPP
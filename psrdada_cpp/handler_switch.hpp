#ifndef PSRDADA_CPP_HANDLER_SWITCH_HPP
#define PSRDADA_CPP_HANDLER_SWITCH_HPP

#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {

template <typename HandlerA, typename HandlerB>
class HandlerSwitch
{
public:
    explicit HandlerSwitch(HandlerA& handler_a, HandlerB& handler_b);
    HandlerSwitch(HandlerSwitch const&) = delete;
    ~HandlerSwitch();
    void init(RawBytes& block);
    bool operator()(RawBytes& block);
    void toggle();
    int handler_idx() const;

private:
    HandlerA& _handler_a;
    HandlerB& _handler_b;
    int _handler_idx;
};

} // psrdada_cpp

#include "psrdada_cpp/detail/handler_switch.cpp"

#endif //PSRDADA_CPP_HANDLER_SWITCH_HPP
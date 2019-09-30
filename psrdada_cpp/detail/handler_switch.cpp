#include "psrdada_cpp/handler_switch.hpp"

template <typename HandlerA, typename HandlerB>
HandlerSwitch<HandlerA, HandlerB>::HandlerSwitch(HandlerA& handler_a, HandlerB& handler_b)
: _handler_a(handler_a)
, _handler_b(handler_b)
, _handler_idx(0)
{
}

template <typename HandlerA, typename HandlerB>
HandlerSwitch<HandlerA, HandlerB>::~HandlerSwitch()
{
}


template <typename HandlerA, typename HandlerB>
void HandlerSwitch<HandlerA, HandlerB>::init(RawBytes& block)
{
    _handler_a.init(block);
    _handler_b.init(block);
}

template <typename HandlerA, typename HandlerB>
bool HandlerSwitch<HandlerA, HandlerB>::operator()(RawBytes& block)
{
    if (_handler_idx == 0)
    {
        return _handler_a(block);
    }
    else
    {
        return _handler_b(block);
    }
}

template <typename HandlerA, typename HandlerB>
void HandlerSwitch<HandlerA, HandlerB>::toggle()
{
    if (_handler_idx == 0)
    {
        _handler_idx = 1;
    }
    else
    {
        _handler_idx = 0;
    }
}

template <typename HandlerA, typename HandlerB>
int HandlerSwitch<HandlerA, HandlerB>:: handler_idx() const
{
    return _handler_idx;
}

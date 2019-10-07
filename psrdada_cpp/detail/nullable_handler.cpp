#include "psrdada_cpp/nullable_handler.hpp"

template <typename Handler>
NullableHandler<Handler>::NullableHandler(Handler& handler)
: HandlerSwitch<Handler, NullSink>(handler, _null)
{
}

template <typename Handler>
NullableHandler<Handler>::~NullableHandler()
{
}

template <typename Handler>
bool NullableHandler<Handler>::is_nulled() const
{
    return (this->handler_idx() == 1);
}


template <typename Handler>
void NullableHandler<Handler>::unnull()
{
    if (is_nulled())
    {
        this->toggle();
    }
}

template <typename Handler>
void NullableHandler<Handler>::null()
{
    if (!is_nulled())
    {
        this->toggle();
    }
}
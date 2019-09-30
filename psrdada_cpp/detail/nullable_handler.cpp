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

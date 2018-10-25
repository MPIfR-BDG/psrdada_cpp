namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

template <>
long double Header::get<long double>(char const* key)
{
    fetch_header_string(key);
    long double value = std::strtold(_buffer, NULL);
    BOOST_LOG_TRIVIAL(info) << key << " = " << value;
    return value;
}

template <>
std::size_t Header::get<std::size_t>(char const* key)
{
    fetch_header_string(key);
    std::size_t value = std::strtoul(_buffer, NULL);
    BOOST_LOG_TRIVIAL(info) << key << " = " << value;
    return value;
}

template <>
long double Header::set<long double>(char const* key, T value)
{
    ascii_header_set(_header.ptr(), key, "%ld", value);
}

template <>
std::size_t Header::set<std::size_t>(char const* key, T value)
{
    ascii_header_set(_header.ptr(), key, "%ul", value);
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
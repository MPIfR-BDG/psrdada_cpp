#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include <string>


namespace psrdada_cpp
{
    
    template<typename NumericT>
    void SigprocHeader::header_write(char*& ptr, std::string const& name, NumericT val)
    {
        header_write(ptr,name);
        std::memcpy(ptr,(char*)&val,sizeof(val));
        ptr += sizeof(val);
    }

} // namespace psrdada_cpp 

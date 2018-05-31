#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include <string>


namespace psrdada_cpp
{
    
    template<class String>
    void SigprocHeader::header_write(RawBytes& block, String str)
    {
        std::string s = str;
        int len = s.size();
        std::copy((char*)&len,(char*)&len + sizeof(len),block.ptr());
        std::copy(str.begin(),str.end(),block.ptr());
    }
    
    template<class String, typename NumericT>
    void SigprocHeader::header_write(RawBytes& block, String name, NumericT val)
    {
        header_write<String>(block,name);
        std::copy((char*)&val, (char*)&val + sizeof(val),block.ptr());
    }

} // namnespace psrdada_cpp 

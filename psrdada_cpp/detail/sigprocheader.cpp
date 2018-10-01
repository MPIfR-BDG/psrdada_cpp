#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include <string>


namespace psrdada_cpp
{
    
    template<class String>
    void SigprocHeader::header_write(char*& ptr, String str)
    {
        std::string s = str;
        int len = s.size();
        std::memcpy(ptr,(char*)&len,sizeof(len));
	ptr += sizeof(len);
        std::copy(str.begin(),str.end(),ptr);
	ptr += sizeof(str);
    }
    
    template<class String, typename NumericT>
    void SigprocHeader::header_write(char*& ptr, String name, NumericT val)
    {
        header_write<String>(ptr,name);
        std::memcpy(ptr,(char*)&val,sizeof(val));
	ptr += sizeof(val);
    }

} // namespace psrdada_cpp 

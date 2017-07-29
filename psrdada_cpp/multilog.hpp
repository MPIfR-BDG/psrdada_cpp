#ifndef PSRDADA_CPP_MULTILOG_HPP
#define PSRDADA_CPP_MULTILOG_HPP

#include <string>
#include "multilog.h"

namespace psrdada_cpp {

    class MultiLog
    {
    private:
        std::string _name;
        multilog_t* _log;
        bool _open;

    public:
        explicit MultiLog(std::string name);
        MultiLog(MultiLog const&) = delete;
        ~MultiLog();

        multilog_t* native_handle();

        template<class... Args>
        void write(int priority, const char* format, Args&&... args);

    private:
        void open();
        void close();
    };

    template<class... Args>
    void MultiLog::write(int priority, const char* format, Args&&... args)
    {
        if (!_open)
        {
            throw std::runtime_error("MultiLog must be opened before writing");
        }
        multilog(_log, format, std::forward<Args>(args)...);
    }

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MULTILOG_HPP
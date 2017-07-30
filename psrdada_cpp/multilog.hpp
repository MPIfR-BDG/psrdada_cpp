#ifndef PSRDADA_CPP_MULTILOG_HPP
#define PSRDADA_CPP_MULTILOG_HPP

#include "multilog.h"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    /**
     * @brief      A class for wrapping multilog_t instances
     *             required for logging with the underlying
     *             DADA API.
     */
    class MultiLog
    {
    private:
        std::string _name;
        multilog_t* _log;
        bool _open;

    public:
        /**
         * @brief      Create a new instance
         *
         * @param[in]  name  The name to give this logger
         */
        explicit MultiLog(std::string name);
        MultiLog(MultiLog const&) = delete;
        ~MultiLog();

        /**
         * @brief      Get a native handle to the wrapped multilog_t pointer
         */
        multilog_t* native_handle();

        /**
         * @brief      Write to the log
         *
         * @param[in]  priority   The priority (0, 1, 2...)
         * @param[in]  format     The format string
         * @param[in]  ...        Parameters for the format string
         *
         * @tparam     Args       The types of the parameters for the format string
         */
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
        multilog(_log, priority, format, std::forward<Args>(args)...);
    }

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MULTILOG_HPP
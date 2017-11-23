#ifndef PSRDADA_CPP_MEERKAT_TOOLS_FENG_HEADER_INSERTER_HPP
#define PSRDADA_CPP_MEERKAT_TOOLS_FENG_HEADER_INSERTER_HPP

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include <cstdint>
#include <time.h>
#include <cmath>

namespace psrdada_cpp {
namespace meerkat {
namespace tools {

template <typename Handler>
class FengHeaderInserter
{
    public:
        typedef int64_t timestamp_t;

    public:
        explicit FengHeaderInserter(
            Handler const& handler,
            std::string const& obs_id,
            float cfreq,
            float bw,
            std::size_t nchans,
            double sync_epoch);
        ~FengHeaderInserter();

        void init(RawBytes& block);
        bool operator()(RawBytes& block);

    private:
        void calculate_epoch(timestamp_t timestamp);

    private:
        Handler const& _handler;
        std::string const& _obs_id;
        float _cfreq;
        float _bw;
        std::size_t _nchans;
        double _sync_epoch;
        char _utc_string[64];
};

} //tools
} //meerkat
} //psrdada_cpp

#include "psrdada_cpp/meerkat/tools/detail/feng_header_inserter.cu"

#endif //PSRDADA_CPP_MEERKAT_TOOLS_FENG_HEADER_INSERTER_HPP
#include "psrdada_cpp/dada_client_base.hpp"
#include <exception>

namespace psrdada_cpp {

    DadaClientBase::DadaClientBase(key_t key, MultiLog const& log)
    : _key(key)
    , _log(log)
    {
        connect();
    }

    DadaClientBase::~DadaClientBase()
    {
        if (_connected)
        {
            disconnect();
        }
    }

    std::size_t DadaClientBase::data_buffer_size()
    {
        return ipcbuf_get_bufsz((ipcbuf_t *) _hdu->data_block);
    }

    std::size_t DadaClientBase::header_buffer_size()
    {
        return ipcbuf_get_bufsz (_hdu->header_block);
    }

    void DadaClientBase::connect()
    {
        _hdu = dada_hdu_create(_log.native_handle());
        dada_hdu_set_key(_hdu, _key);
        if (dada_hdu_connect (_hdu) < 0){
            _log.write(LOG_ERR, "could not connect to hdu\n");
            throw std::runtime_error("Unable to connect to hdu\n");
        }
        _connected = true;
    }

    void DadaClientBase::disconnect()
    {
        if (dada_hdu_disconnect (_hdu) < 0){
            _log.write(LOG_ERR, "could not disconnect from hdu\n");
            throw std::runtime_error("Unable to disconnect from hdu\n");
        }
        dada_hdu_destroy(_hdu);
        _connected = false;
    }

} //namespace psrdada_cpp
#include "psrdada_cpp/dada_client_base.hpp"

namespace psrdada_cpp {

    DadaClientBase::DadaClientBase(key_t key, MultiLog& log)
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

    std::size_t DadaClientBase::data_buffer_size() const
    {
        return ipcbuf_get_bufsz((ipcbuf_t *) _hdu->data_block);
    }

    std::size_t DadaClientBase::header_buffer_size() const
    {
        return ipcbuf_get_bufsz(_hdu->header_block);
    }

    std::size_t DadaClientBase::data_buffer_count() const
    {
        return ipcbuf_get_nbufs((ipcbuf_t *) _hdu->data_block);
    }

    std::size_t DadaClientBase::header_buffer_count() const
    {
        return ipcbuf_get_nbufs(_hdu->header_block);
    }

    void DadaClientBase::connect()
    {
        BOOST_LOG_TRIVIAL(debug) << "Connecting to dada buffer ["
        << std::hex << _key << std::dec << "]";
        _hdu = dada_hdu_create(_log.native_handle());
        dada_hdu_set_key(_hdu, _key);
        if (dada_hdu_connect (_hdu) < 0){
            _log.write(LOG_ERR, "could not connect to hdu\n");
            throw std::runtime_error("Unable to connect to hdu\n");
        }
        BOOST_LOG_TRIVIAL(debug) << "Header buffer is " << header_buffer_count()
            << " x " << header_buffer_size() << " bytes";
        BOOST_LOG_TRIVIAL(debug) << "Data buffer is " << data_buffer_count()
            << " x " << data_buffer_size() << " bytes";
        _connected = true;
    }

    void DadaClientBase::disconnect()
    {
        BOOST_LOG_TRIVIAL(debug) << "Disconnecting from dada buffer ["
        << std::hex << _key << std::dec << "]";
        if (dada_hdu_disconnect (_hdu) < 0){
            _log.write(LOG_ERR, "could not disconnect from hdu\n");
            throw std::runtime_error("Unable to disconnect from hdu\n");
        }
        dada_hdu_destroy(_hdu);
        _connected = false;
    }

    void DadaClientBase::reconnect()
    {
        disconnect();
        connect();
    }

} //namespace psrdada_cpp
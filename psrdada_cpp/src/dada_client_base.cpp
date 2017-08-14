#include "psrdada_cpp/dada_client_base.hpp"

namespace psrdada_cpp {

    DadaClientBase::DadaClientBase(key_t key, MultiLog& log)
    : _key(key)
    , _log(log)
    {
        connect();
        std::stringstream _key_string_stream;
        _key_string_stream << "["<< std::hex << _key << std::dec << "] ["<<_log.name()<<"] ";
        _id = _key_string_stream.str();
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
        BOOST_LOG_TRIVIAL(debug) << this->id() << "Connecting to dada buffer";
        _hdu = dada_hdu_create(_log.native_handle());
        dada_hdu_set_key(_hdu, _key);
        if (dada_hdu_connect (_hdu) < 0){
            _log.write(LOG_ERR, "could not connect to hdu\n");
            throw std::runtime_error("Unable to connect to hdu\n");
        }
        BOOST_LOG_TRIVIAL(debug) << this->id() << "Header buffer is " << header_buffer_count()
            << " x " << header_buffer_size() << " bytes";
        BOOST_LOG_TRIVIAL(debug) << this->id() << "Data buffer is " << data_buffer_count()
            << " x " << data_buffer_size() << " bytes";
        _connected = true;
    }

    void DadaClientBase::disconnect()
    {
        BOOST_LOG_TRIVIAL(debug) << this->id() << "Disconnecting from dada buffer";
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

    std::string const& DadaClientBase::id() const
    {
        return _id;
    }


} //namespace psrdada_cpp
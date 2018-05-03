#include "psrdada_cpp/dada_write_client.hpp"

namespace psrdada_cpp {

    DadaWriteClient::DadaWriteClient(key_t key, MultiLog& log)
    : DadaClientBase(key, log)
    , _locked(false)
    , _header_stream(*this)
    , _data_stream(*this)
    {
        lock();
    }

    DadaWriteClient::~DadaWriteClient()
    {
        release();
    }

    void DadaWriteClient::lock()
    {
        if (!_connected)
        {
            throw std::runtime_error("Lock requested on unconnected HDU\n");
        }
        BOOST_LOG_TRIVIAL(debug) << this->id() << "Acquiring writing lock on dada buffer";
        if (dada_hdu_lock_write (_hdu) < 0)
        {
            _log.write(LOG_ERR, "open_hdu: could not lock write\n");
            throw std::runtime_error("Error locking HDU");
        }
        _locked = true;
    }

    void DadaWriteClient::release()
    {
        if (!_locked)
        {
            throw std::runtime_error("Release requested on unlocked HDU\n");
        }
        BOOST_LOG_TRIVIAL(debug) << this->id() << "Releasing writing lock on dada buffer";
        if (dada_hdu_unlock_write (_hdu) < 0)
        {
            _log.write(LOG_ERR, "open_hdu: could not release write\n");
            throw std::runtime_error("Error releasing HDU");
        }
        _locked = false;
    }

    DadaWriteClient::HeaderStream& DadaWriteClient::header_stream()
    {
        return _header_stream;
    }

    DadaWriteClient::DataStream& DadaWriteClient::data_stream()
    {
        return _data_stream;
    }

    void DadaWriteClient::reset()
    {
        release();
        reconnect();
        lock();
    }

    DadaWriteClient::HeaderStream::HeaderStream(DadaWriteClient& parent)
    : _parent(parent)
    , _current_block(nullptr)
    {
    }

    DadaWriteClient::HeaderStream::~HeaderStream()
    {
    }

    RawBytes& DadaWriteClient::HeaderStream::next()
    {
        if (_current_block)
        {
            throw std::runtime_error("Previous header block not released");
        }
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Acquiring next header block";
        char* tmp = ipcbuf_get_next_write(_parent._hdu->header_block);
        _current_block.reset(new RawBytes(tmp, _parent.header_buffer_size()));
        return *_current_block;
    }

    void  DadaWriteClient::HeaderStream::release()
    {
        if (!_current_block)
        {
            throw std::runtime_error("No header block to be released");
        }
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Writing header content:\n " << _current_block->ptr();
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Header bytes used " << _current_block->used_bytes();
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Releasing header block";
        if (ipcbuf_mark_filled(_parent._hdu->header_block, _current_block->used_bytes()) < 0)
        {
            _parent._log.write(LOG_ERR, "Could not mark filled header block\n");
            throw std::runtime_error("Could not mark filled header block");
        }
        _current_block.reset();
    }

    DadaWriteClient::DataStream::DataStream(DadaWriteClient& parent)
    : _parent(parent)
    , _current_block(nullptr)
    , _block_idx(0)
    {
    }

    DadaWriteClient::DataStream::~DataStream()
    {
    }

    RawBytes& DadaWriteClient::DataStream::next()
    {
        if (_current_block)
        {
            throw std::runtime_error("Previous data block not released");
        }
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Acquiring next header block";
        char* tmp = ipcio_open_block_write(_parent._hdu->data_block, &_block_idx);
        _current_block.reset(new RawBytes(tmp, _parent.data_buffer_size()));
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Acquired data block " << _block_idx;
        return *_current_block;
    }

    void DadaWriteClient::DataStream::release(bool eod)
    {
        if (!_current_block)
        {
             throw std::runtime_error("No data block to be released");
        }
        BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Releasing data block";
        if (eod)
        {
            BOOST_LOG_TRIVIAL(debug) << _parent.id() << "Setting EOD markers";
            if (ipcio_update_block_write (_parent._hdu->data_block, _current_block->used_bytes()) < 0)
            {
                _parent._log.write(LOG_ERR, "close_buffer: ipcio_update_block_write failed\n");
                throw std::runtime_error("Could not close ipcio data block");
            }
        }
        else
        {
            if (ipcio_close_block_write (_parent._hdu->data_block, _current_block->used_bytes()) < 0)
            {
                _parent._log.write(LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
                throw std::runtime_error("Could not close ipcio data block");
            }
            _current_block.reset();
        }
    }

    std::size_t DadaWriteClient::DataStream::block_idx() const
    {
        return _block_idx;
    }

} //namespace psrdada_cpp






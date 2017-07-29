#include "psrdada_cpp/dada_write_client.hpp"

namespace psrdada_cpp {

    DadaWriteClient::DadaWriteClient(key_t key, MultiLog& log)
    : DadaClientBase(key, log)
    , _locked(false)
    , _current_header_block(nullptr)
    , _current_data_block(nullptr)
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
            throw std::runtime_error("Lock requested on unconnected HDU\n");
        if (dada_hdu_lock_write (_hdu) < 0)
        {
            _log.write(LOG_ERR, "open_hdu: could not lock write\n");
            throw std::runtime_error("Error locking HDU");
        }
        _locked = true;
    }

    bool DadaWriteClient::is_locked() const
    {
        return _locked;
    }

    void DadaWriteClient::release()
    {
       if (!_locked)
            throw std::runtime_error("Release requested on unlocked HDU\n");
        if (dada_hdu_unlock_write (_hdu) < 0)
        {
            _log.write(LOG_ERR, "open_hdu: could not release write\n");
            throw std::runtime_error("Error releasing HDU");
        }
        _locked = false;
    }

    RawBytes& DadaWriteClient::acquire_header_block()
    {
        if (_current_header_block)
        {
            throw std::runtime_error("Previous header block not released");
        }
        char* tmp = ipcbuf_get_next_write(_hdu->header_block);
        _current_header_block.reset(new RawBytes(tmp, header_buffer_size()));
        return *_current_header_block;
    }

    void  DadaWriteClient::release_header_block()
    {
        if (!_current_header_block)
        {
            throw std::runtime_error("No header block to be released");
        }

        if (ipcbuf_mark_filled(_hdu->header_block, _current_header_block->used_bytes()) < 0)
        {
            _log.write(LOG_ERR, "Could not mark filled header block\n");
            throw std::runtime_error("Could not mark filled header block");
        }
        _current_header_block.reset(nullptr);
    }

    RawBytes& DadaWriteClient::acquire_data_block()
    {
        if (_current_data_block)
        {
            throw std::runtime_error("Previous data block not released");
        }
        std::size_t block_idx = 0;
        char* tmp = ipcio_open_block_write(_hdu->data_block, &block_idx);
        _current_data_block.reset(new RawBytes(tmp,data_buffer_size()));
        return *_current_data_block;
    }

    void DadaWriteClient::release_data_block(bool eod)
    {
        if (!_current_data_block)
        {
             throw std::runtime_error("No data block to be released");
        }
        if (eod)
        {
            if (ipcio_update_block_write (_hdu->data_block, _current_data_block->used_bytes()) < 0)
            {
                _log.write(LOG_ERR, "close_buffer: ipcio_update_block_write failed\n");
                throw std::runtime_error("Could not close ipcio data block");
            }
        }
        else
        {
            if (ipcio_close_block_write (_hdu->data_block, _current_data_block->used_bytes()) < 0)
            {
                _log.write(LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
                throw std::runtime_error("Could not close ipcio data block");
            }
            _current_data_block.reset(nullptr);
        }
    }

} //namespace psrdada_cpp






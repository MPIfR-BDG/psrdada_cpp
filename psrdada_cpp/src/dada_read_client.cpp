#include "psrdada_cpp/dada_read_client.hpp"

namespace psrdada_cpp {

    DadaReadClient::DadaReadClient(key_t key, MultiLog const& log)
    : DadaClientBase(key, log)
    , _locked(false)
    , _current_header_block(nullptr)
    , _current_data_block(nullptr)
    , _current_data_block_idx(0)
    {
        lock();
    }

    DadaReadClient::~DadaReadClient()
    {
        release();
    }

    void DadaReadClient::lock()
    {
        if (!_connected)
            throw std::runtime_error("Lock requested on unconnected HDU\n");
        if (dada_hdu_lock_read (_hdu) < 0)
        {
            _log.write(LOG_ERR, "open_hdu: could not lock read\n");
            throw std::runtime_error("Error locking HDU");
        }
        _locked = true;
    }

    void DadaReadClient::release()
    {
       if (!_locked)
            throw std::runtime_error("Release requested on unlocked HDU\n");
        if (dada_hdu_unlock_read (_hdu) < 0)
        {
            _log.write(LOG_ERR, "open_hdu: could not release read\n");
            throw std::runtime_error("Error releasing HDU");
        }
        _locked = false;
    }

    RawBytes& DadaReadClient::acquire_header_block()
    {
        if (_current_header_block)
        {
            throw std::runtime_error("Previous header block not released");
        }
        std::size_t nbytes = 0;
        char* tmp = ipcbuf_get_next_read(_hdu->header_block, &nbytes);
        if (!tmp)
        {
            _log.write(LOG_ERR, "Could not get header\n");
            throw std::runtime_error("Could not get header");
        }
        _current_header_block.reset(new RawBytes(tmp, header_buffer_size(), nbytes));
        return *_current_header_block;
    }

    void  DadaReadClient::release_header_block()
    {
        if (!_current_header_block)
        {
            throw std::runtime_error("No header block to be released");
        }

        if (ipcbuf_mark_cleared(_hdu->header_block) < 0)
        {
            _log.write(LOG_ERR, "Could not mark cleared header block\n");
            throw std::runtime_error("Could not mark cleared header block");
        }
        _current_header_block.reset(nullptr);
    }

    RawBytes& DadaReadClient::acquire_data_block()
    {
        if (_current_data_block)
        {
            throw std::runtime_error("Previous data block not released");
        }
        std::size_t nbytes = 0;
        char* tmp = ipcio_open_block_read(_hdu->data_block, &nbytes, &_current_data_block_idx);
        if (!tmp)
        {
            _log.write(LOG_ERR, "Could not get data block\n");
            throw std::runtime_error("Could not open block to read");
        }
        _current_data_block.reset(new RawBytes(tmp, data_buffer_size(), nbytes));
    }

    void DadaReadClient::release_data_block()
    {
        if (!_current_data_block)
        {
             throw std::runtime_error("No data block to be released");
        }
        if (ipcio_close_block_read (_hdu->data_block, _current_data_block->used_bytes()) < 0)
        {
            log.write(LOG_ERR, "close_buffer: ipcio_close_block_read failed\n");
            throw std::runtime_error("Could not close ipcio data block");
        }
        _current_data_block.reset(nullptr);
    }

    std::size_t DadaReadClient::current_data_block_idx()
    {
        if (!_current_data_block)
        {
             throw std::runtime_error("No data block currently acquired");
        }
        return _current_data_block_idx
    }

} //namespace psrdada_cpp






#include "psrdada_cpp/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <ctime>
#include <thread>
namespace psrdada_cpp {

    template <class HandlerType>
    TransposeToDada<HandlerType>::TransposeToDada(std::size_t numbeams, std::vector<std::shared_ptr<HandlerType>> handler)
    : _numbeams(numbeams)
    , _handler(std::move(handler))
    , _nchans(128)
    , _nsamples(64)
    , _ntime(64)
    , _nfreq(32)
    , _ngroups(10)
    {
    }

    template <class HandlerType>
    TransposeToDada<HandlerType>::~TransposeToDada()
    {
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::init(RawBytes& block)
    {
        std::uint32_t ii;
        for (ii = 0; ii < _numbeams; ii++ )
        {
            (*_handler[ii]).init(block);
        }
    }

    template <class HandlerType>
    bool TransposeToDada<HandlerType>::operator()(RawBytes& block)
    {
    	  
        std::uint32_t ii;
	std::vector<std::thread> threads;
        for(ii=0; ii< _numbeams; ii++)
        {
		threads.emplace_back(std::thread([&]()
		{
		char* o_data = new char[_nchans*_nsamples*_ntime*_nfreq*_ngroups];
		RawBytes transpose(o_data,std::size_t(_nchans*_nsamples*_ntime*_nfreq*_ngroups),std::size_t(0));
		//clock_t st = clock();
		for (std::uint32_t jj=202375100; jj < 202375200; ++jj)
		{
		    std::cout << (int)block.ptr()[jj] << "\n";
		}
		transpose::do_transpose(transpose,block,_nchans,_nsamples,_ntime,_nfreq,ii,_numbeams,_ngroups);
		//clock_t spt = clock();
		//std::cout << "Time taken:" << (double)(spt -st)/CLOCKS_PER_SEC << "\n";
		transpose.used_bytes(transpose.total_bytes());
		(*_handler[ii])(transpose);
		}));
		
	}

	for (ii=0; ii< _numbeams; ii++)
	{
	        threads[ii].join();
	}

        return false;
    }
 
    template <class HandlerType>
    void TransposeToDada<HandlerType>::set_nchans(const int nchans)
    {
        _nchans = nchans;
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::set_nbeams(const int nbeams)
    {
        _numbeams = nbeams;
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::set_ngroups(const int ngroups)
    {
        _ngroups = ngroups;
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::set_ntime(const int ntime)
    {
        _ntime = ntime;
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::set_nsamples(const int nsamples)
    {
        _nsamples = nsamples;
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::set_nfreq(const int nfreq)
    {
        _nfreq = nfreq;
    }

    template <class HandlerType>
    std::uint32_t TransposeToDada<HandlerType>::nchans()
    {
        return _nchans;
    }

    template <class HandlerType>
    std::uint32_t TransposeToDada<HandlerType>::nsamples()
    {
        return _nsamples;
    }

    template <class HandlerType>
    std::uint32_t TransposeToDada<HandlerType>::ntime()
    {
        return _ntime;
    }

    template <class HandlerType>
    std::uint32_t TransposeToDada<HandlerType>::nfreq()
    {
        return _nfreq;
    }

    template <class HandlerType>
    std::uint32_t TransposeToDada<HandlerType>::nbeams()
    {
	return _numbeams;
    }

    template <class HandlerType>
    std::uint32_t TransposeToDada<HandlerType>::ngroups()
    {
	return _ngroups;
    }

} //psrdada_cpp

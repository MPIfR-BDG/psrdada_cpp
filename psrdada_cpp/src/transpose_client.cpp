#include "psrdada_cpp/transpose_client.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/multilog.hpp"

namespace psrdada_cpp {

    TransposeClient::TransposeClient(DadaWriteClient* _writer[],std::string* keys,std::uint32_t numbeams)
    : _nbeams(numbeams),
      _nchans(128),
      _nsamples(128),
      _ntime(64),
      _nfreq(32)
    {
        std::uint32_t ii;
        std::string logname="logger";
        MultiLog log(logname);
        _writer = new DadaWriteClient*[numbeams];
        key_t key;
        for (ii=0 ; ii< numbeams; ii ++)
        {
            key = string_to_key(*(keys + ii));
            *(_writer + ii) = new DadaWriteClient(key, log)  ;
        }

    }

    TransposeClient::~TransposeClient()
    {
    }

    /**
     * @brief: The method that transposes per beam. User needs to initiate multiple threads corresponding to multiple beams
     */


    void TransposeClient::do_transpose(RawBytes& input_data, RawBytes& transposed_data,std::uint32_t beamnum)
    {


        // For now making a separate output memory block in RAM and constructing a RawBytes object pointing to it
	// Can be done more memory efficiently by using a temporary pointer
	

        std::uint32_t j,k,l,m;
	std::uint32_t a = 0;

        // Actual transpose

	    for (j =0; j < _nsamples; j++)
            {
                for (k = 0; k < _ntime ; k++)
		{

		    for (l = 0; l < _nfreq ; l++)
		    {

										                         
		        for (m=0;m < _nchans ; m++)
		        {

		            transposed_data.ptr()[a] = input_data.ptr()[m + _ntime * _nchans * _nsamples * l + _nchans * (j * _ntime + k) + _nsamples * _nchans * _ntime* _nfreq * beamnum];
			    a++;

		        }
					                            																		                                                                             
		    }

																						
	        }                                                                                                       
                                                                                                      
	    }
	 
	
    }

   /* @brief: Function that reads in the data from the DADA buffer adn returns a Reference to a RawBytes object */


    RawBytes& TransposeClient::read_to_transpose(DadaReadClient& _reader)
    {
        // Get the current block of data
        DadaReadClient::DataStream& data_stream = _reader.data_stream();
        // Get the RawBytes object pointing to the current block of memory
        auto& current_block = data_stream.next();
       // Release the current block of data
        data_stream.release();
	return current_block;
    }


    /* @brief: Method to write the block of transposed data to a DADA buffer.*/

    void TransposeClient::write_transpose(RawBytes& transposed_data, DadaWriteClient& writer)
    {
        DadaWriteClient::DataStream& data_stream = writer.data_stream();
	auto& current_block = data_stream.next();
	//Write the transposed data(std::memcpy??)
	std::memcpy(current_block.ptr(), transposed_data.ptr(),transposed_data.used_bytes());
        current_block.used_bytes(transposed_data.used_bytes());
	data_stream.release();

    }


    /* All the setters for setting up the transpose client block */

    void TransposeClient::set_nbeams(const int nbeams)
    {
         _nbeams = nbeams;
    }

    void TransposeClient::set_nchans(const int nchans)
    {
        _nchans = nchans;
    }

    void TransposeClient::set_ntime(const int ntime)
    {
        _ntime = ntime;
    }

    void TransposeClient::set_nsamples(const int nsamples)
    {
        _nsamples = nsamples;
    }

    void TransposeClient::set_nfreq(const int nfreq)
    {
        _nfreq = nfreq;
    }


    std::uint32_t TransposeClient::nbeams()
    {
        return _nbeams;
    }

    std::uint32_t TransposeClient::nchans()
    {
        return _nchans;
    }

    std::uint32_t TransposeClient::nsamples()
    {
        return _nsamples;
    }

    std::uint32_t TransposeClient::ntime()
    {
        return _ntime;
    }

    std::uint32_t TransposeClient::nfreq()
    {
        return _nfreq;
    }

} //namespace psrdada_cpp

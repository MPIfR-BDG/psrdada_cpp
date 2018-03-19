#ifndef PSRDADA_CPP_TRANSPOSE_CLIENT_HPP
#define PSRDADA_CPP_TRANSPOSE_CLIENT_HPP

#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"


namespace psrdada_cpp {

    /**
     * @brief      Class that provides means for doing transpose
     *             by reading data from a DADA ring buffer and 
     *             write out the transposed data to a DADA ring
     *             buffer
     */
    class TransposeClient 
    {
    public:

        TransposeClient(DadaWriteClient* _writer[],std::string* keys, std::uint32_t numbeams);
        TransposeClient(TransposeClient const&) = delete;
	~TransposeClient();

        /**
	 * @brief function to the do the transpose
	 */
        void do_transpose(RawBytes& _current_block,RawBytes& transposed_data, std::uint32_t beamnum); 
        
	/**
	 * @brief Function to read in the data to be transposed
	 */
        RawBytes& read_to_transpose(DadaReadClient& _reader);

        /**
	 * @brief Function to write transposed data to the dada buffer
	 */
        void write_transpose(RawBytes& transposed_data, DadaWriteClient& writer);


       /**
        * @brief Setter for the numbeams
        */

        void set_nbeams(const int nbeams);   
     
       /**
        * @brief Setter for frequency channels
        */

        void set_nchans(const int nchans);

       /**
        * @brief Setter of number of samples
        */

        void set_nsamples(const int nsamples);
  
       /**
        * @brief Setter for number of time samples
        */

        void set_ntime(const int ntime);

       /**
        * @brief Setter for number of frequency blocks
        */

        void set_nfreq(const int _nfreq);

       /**
        * @ getter for number of beams
        */

        std::uint32_t nbeams();
 
       /**
        * @ getter for number of channels
        */

        std::uint32_t nchans();

       /**
        * @ getter for number of time blocks
        */

        std::uint32_t nsamples();

       /**
        * @ getter for number of time samples
        */

        std::uint32_t ntime();

       /**
        * @ getter for number of freq blocks
        */

        std::uint32_t nfreq();
 

    private:
        std::uint32_t _nbeams;
	std::uint32_t _nchans;
	std::uint32_t _nsamples;
	std::uint32_t _ntime;
	std::uint32_t _nfreq;

    };
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_TRANSPOSE_CLIENT_HPP

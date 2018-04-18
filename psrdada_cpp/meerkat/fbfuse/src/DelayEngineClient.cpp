#include <boost/tokenizer.hpp>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace detail {

void get_tokens(std::string const& input, std::vector<std::string>& tokens)
{
    tokens.clear();
    boost::char_separator<char> sep(" ", "");
    boost::tokenizer<boost::char_separator<char>> tok(input, sep);
    for(auto beg=tok.begin(); beg!=tok.end();++beg){
        tokens.push_back(*beg);
    }
}

}

DelayEngineClient::DelayEngineClient(DelayEngineConfig const& config)
{

}

DelayEngineClient::~DelayEngineClient()
{

}

void DelayEngineClient::connect()
{
    _socket.reset(new tcp::socket(_io_service));
    try
    {
        tcp::resolver resolver(_io_service);
        tcp::resolver::query query(tcp::v4(), _delay_engine_address, _delay_engine_port);
        tcp::resolver::iterator iterator = resolver.resolve(query);
        boost::asio::connect(*s, iterator);
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        disconnect();
        throw e;
    }
}

void DelayEngineClient::disconnect()
{
    _socket.reset(nullptr);
}

void DelayEngineClient::recv_katcp_version()
{
    boost::asio::streambuf buffer;
    std::string response;
    std::istream is(&buffer);
    for (int ii=0; ii < KATCP_VERSION_STRING_COUNT; ++ii)
    {
        boost::asio::read_until(*_socket, buffer, KATCP_REPLY_TERMINATOR);
        std::getline(is, response);
    }
}

void DelayEngineClient::send_delay_request(time_t start_epoch)
{
    std::stringstream request;
    ss << "?delays " << _delay_engine_key << " " << epoch << KATCP_REQUEST_TERMINATOR;
    boost::asio::write(*_socket, boost::asio::buffer(request.str().c_str(),req.str().size()));
}

void DelayEngineClient::recv_delay_response(DelayModel& model)
{

    /*
     * The format of the delay response take the form:
     *
     * #delays <start time UTC seconds> <end time UTC seconds>
     * #delays <beam ID> <antenna ID> <rate> <offset>
     * #delays <beam ID> <antenna ID> <rate> <offset>
     * ...
     * #delays <beam ID> <antenna ID> <rate> <offset>
     * !delays <number of delay updates> ok
     *
     * In the case of a server side error the reply message will
     * say:
     * !delays fail <reason for failure>
     *
     */

    std::lock_guard<std::mutex> lock(model.mutex());

    boost::asio::streambuf buffer;
    std::string response;
    std::istream is(&buffer);
    std::vector<std::string> tokens;

    // First we receive the validity epochs
    boost::asio::read_until(*_socket, buffer, KATCP_REPLY_TERMINATOR);
    std::getline(is, response);
    get_tokens(response, tokens);
    if (tokens.size() != VALIDITY_MESSAGE_TOKENS)
    {
        //Warning unexpected response;
    }
    else
    {
        //double tstart   = std::strtod(token[1],NULL);
        //double tfinish = std::strtod(token[2],NULL);
        //model.set_validity(tstart,tfinish);
    }

    //expected number of updates is what?

    // Next receive all delay updates
    for (int ii=0; ii<number_of_updates; ++ii)
    {
        boost::asio::read_until(*_socket, buffer, KATCP_REPLY_TERMINATOR);
        std::getline(is, response);
        get_tokens(response, tokens);

        if (response.compare(0, 1, KATCP_REPLY_PREFIX)==0)
        {
            //error, reply message before all updates received
            //raise errir with contents of reply
        }
        else if (response.compare(0, 1, KATCP_INFORM_PREFIX)==0)
        {
            if (tokens.size() != DELAY_MESSAGE_TOKENS)
            {
                //Error unexpected number of tokens in message
            }
            else
            {
                //beam    = token[1];
                //antenna = token[2];
                //delay_rate   = std::strtod(token[3],NULL);
                //delay_offset = std::strtod(token[4],NULL);
                //model.update(beam, antenna, coeff0, coeff1);
            }
        }
    }

    //Finally receive the reply
    boost::asio::read_until(*_socket, buffer, KATCP_REPLY_TERMINATOR);
    std::getline(is, response);
    get_tokens(response, tokens);
    if (response.compare(0, 1, KATCP_REPLY_PREFIX)!=0)
    {
        //error, expected reply message
    }
    else if (tokens[2].compare("ok")==0)
    {
        //Everything is good
    }
    else
    {
        //something wierd right at the end
    }
}


void DelayEngineClient::update_model(DelayModel& model, time_t epoch)
{
    connect();
    recv_katcp_version();
    send_delay_request(epoch);
    recv_delay_response(model);
    disconnect();
}


} //namespace fbfuse
} //namespace meerkat
} //namespace fbfuse
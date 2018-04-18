#ifndef PSRDADA_CPP_MEERKAT_DELAYENGINECLIENT_HPP
#define PSRDADA_CPP_MEERKAT_DELAYENGINECLIENT_HPP


//Need mutex lock for delay model read and update;

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

class DelayEngineClient
{
private:
    std::string _delay_engine_address;
    std::string _delay_engine_port;
    std::string _delay_engine_key;
    boost::asio::io_service _io_service;
    std::scoped_ptr<tcp::socket> _socket;
    bool _first_pass;

    void connect();
    void disconnect();
    void build_request();

public:
    DelayEngineClient(Config const& config, std::string address, std::string port);
    ~DelayEngineClient();

    void update_model(DelayModel& model, time_t tstart, time_t tfinish, time_t tolerance);


};

} //namespace fbfuse
} //namespace meerkat
} //namespace fbfuse

#endif //PSRDADA_CPP_MEERKAT_DELAYENGINECLIENT_HPP
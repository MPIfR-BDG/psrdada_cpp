import logging
import json
import tornado
import signal
import posix_ipc
import ctypes
from threading import Lock
from optparse import OptionParser
from katcp import Sensor, AsyncDeviceServer
from katcp.kattypes import request, return_reply, Int, Str, Discrete

log = logging.getLogger("psrdada_cpp.meerkat.fbfuse.delay_engine_client")

lock = Lock()




"""
The FbfControlServer wraps deployment of the FBFUSE beamformer.
This covers both NUMA nodes on one machine and so the configuration
should be based on the full capabilities of a node.
It performs the following:
    - Registration with the FBFUSE master controller
    - Receives configuration information
    - Initialises necessary DADA buffers
    - JIT compilation of beamformer kernels
    - Starts SPEAD transmitters
        - Needs to know output beam ordering
        - Needs to know output multicast groups
        - Combine this under a beam to multicast map
        - Generate cfg file for transmitter
    - Starts beamformer
    - Starts SPEAD receiver
        - Generate cfg file for receiver that includes fengine order
    - Maintains a share memory buffer with delay polynomials
        - Common to all beanfarmer instances running on the node
    - Samples incoming data stream for monitoring purposes
    - Provides sensors relating to beamformer performance (TBD, could just sample beamformer stdout)
"""

class DelayBufferController(object):
    def __init__(self, delay_engine, nbeams, nantennas, configuration_id, nreaders):
        self._nreaders = nreaders
        self._delay_engine = delay_engine
        self._beam_antenna_map = {}
        self._delays_array = self._delays = np.rec.recarray(nbeams * natennas,
            dtype=[
            ("delay_rate","float32"),("delay_offset","float32")
            ])
        as_bytes = self._delays_array.tobytes()

        self.shared_buffer_key = "{}_delay_buffer".format(configuration_id)
        self.mutex_semaphore_key = "{}_delay_buffer".format(configuration_id)
        self.counting_semaphore_key = "{}_delay_buffer_count".format(configuration_id)

        # This semaphore is required to protect access to the shared_buffer
        # so that it is not read and written simultaneously
        # The value is set to two such that two processes can read simultaneously
        self._mutex_semaphore = posix_ipc.Semaphore(
            self.mutex_semaphore_key,
            flags=sysv_ipc.IPC_CREX,
            initial_value=self._nreaders)

        # This semaphore is used to notify beamformer instances of a change to the
        # delay models. Upon any change its value is simply incremented by one.
        self._counting_semaphore = posix_ipc.Semaphore(self.counting_semaphore_key,
            flags=sysv_ipc.IPC_CREX,
            initial_value=0)

        # This is the share memory buffer that contains the delay models for the
        self._shared_buffer = posix_ipc.SharedMemory(
            self.shared_buffer_key,
            flags=sysv_ipc.IPC_CREX,
            size=len(as_bytes))

    def __del__(self):
        self._shared_buffer.remove()

    def _update(self, rt, t, status, value):
        for model in value.splitlines():
            beam, antenna, rate, offset = model.split(",")
            rate = float(rate)
            offset = float(offset)
            self._delays_array[self._beam_antenna_map[(beam,antenna)]] = (rate, offset)




class FbfControlServer(AsyncDeviceServer):
    VERSION_INFO = ("fbf-control-server-api", 0, 1)
    BUILD_INFO = ("fbf-control-server-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port, de_ip, de_port):
        """
        @brief       Construct new FbfControlServer instance

        @params  ip       The interface address on which the server should listen
        @params  port     The port that the server should bind to
        @params  de_ip    The IP address of the delay engine server
        @params  de_port  The port number for the delay engine server

        """
        self._de_ip = de_ip
        self._de_port = de_port
        self._config_id = None
        self._delay_engine = None
        self._delays = None
        self._shared_delay_buffer = None
        super(FbfControlServer, self).__init__(ip,port)

    def start(self):
        """Start FbfControlServer server"""
        super(FbfControlServer,self).start()
        self._setup_clients()

    def _setup_clients(self):
        self._delay_engine = KATCPClientResource(dict(
            name="delay-engine-server",
            address=(self._de_ip, self._de_port),
            controlled=False))
        self._delay_engine.start()

    def setup_sensors(self):
        """
        @brief    Set up monitoring sensors.

        Sensor list:
        - device-status
        - local-time-synced
        - fbf0-status
        - fbf1-status

        @note     The following sensors are made available on top of defaul sensors
                  implemented in AsynDeviceServer and its base classes.

                  device-status:      Reports the health status of the FBFUSE and associated devices:
                                      Among other things report HW failure, SW failure and observation failure.
        """
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of FBFUSE",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)

    @request(Str(), Str(), Int(), Int(), Int())
    @return_reply()
    def request_configure(self, req, configuration_id, mcast_groups_json, feng_ids_csv,
        beam_feng_ids_csv, nbeams, nchannels, beams_to_mcast_map):
        """
        @brief      Configure FBFUSE to receive and process data from a subarray

        @detail     REQUEST ?configure product_id antennas_csv n_channels streams_json proxy_name
                    Configure FBFUSE for the particular data products

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, which is a useful tag to include
                                      in the data, but should not be analysed further. For example "array_1_bc856M4k".

        @param      antennas_csv      A comma separated list of physical antenna names used in particular sub-array
                                      to which the data products belongs.

        @param      n_channels        The integer number of frequency channels provided by the CBF.

        @param      streams_json      a JSON struct containing config keys and values describing the streams.

                                      For example:

                                      @code
                                         {'stream_type1': {
                                             'stream_name1': 'stream_address1',
                                             'stream_name2': 'stream_address2',
                                             ...},
                                             'stream_type2': {
                                             'stream_name1': 'stream_address1',
                                             'stream_name2': 'stream_address2',
                                             ...},
                                          ...}
                                      @endcode

                                      The steam type keys indicate the source of the data and the type, e.g. cam.http.
                                      stream_address will be a URI.  For SPEAD streams, the format will be spead://<ip>[+<count>]:<port>,
                                      representing SPEAD stream multicast groups. When a single logical stream requires too much bandwidth
                                      to accommodate as a single multicast group, the count parameter indicates the number of additional
                                      consecutively numbered multicast group ip addresses, and sharing the same UDP port number.
                                      stream_name is the name used to identify the stream in CAM.
                                      A Python example is shown below, for five streams:
                                      One CAM stream, with type cam.http.  The camdata stream provides the connection string for katportalclient
                                      (for the subarray that this FBFUSE instance is being configured on).
                                      One F-engine stream, with type:  cbf.antenna_channelised_voltage.
                                      One X-engine stream, with type:  cbf.baseline_correlation_products.
                                      Two beam streams, with type: cbf.tied_array_channelised_voltage.  The stream names ending in x are
                                      horizontally polarised, and those ending in y are vertically polarised.

                                      @code
                                         pprint(streams_dict)
                                         {'cam.http':
                                             {'camdata':'http://10.8.67.235/api/client/1'},
                                          'cbf.antenna_channelised_voltage':
                                             {'i0.antenna-channelised-voltage':'spead://239.2.1.154+4:7148'},
                                          'cbf.coherent_filterbanked_beam':
                                             {'i0.coherent-filterbanked-beam':'spead://239.2.2.150+128:7148'},
                                          'cbf.incoherent_filterbanked_beam':
                                             {'i0.incoherent-filterbanked-beam':'spead://239.2.3.150:7148'},
                                          ...}
                                      @endcode

                                      If using katportalclient to get information from CAM, then reconnect and re-subscribe to all sensors
                                      of interest at this time.

        @note       A configure call will result in the generation of a new subarray instance in FBFUSE that will be added to the clients list.

        @return     katcp reply object [[[ !configure ok | (fail [error description]) ]]]
        """
        @tornado.gen.coroutine
        def configure():
            """
            Tasks:
                - compile kernels
                - create shared memory banks
            """
            # Parse feng ids
            feng_ids = [int(idx) for idx in feng_ids_csv.split(",")]
            beam_feng_ids = [int(idx) for idx in beam_feng_ids_csv.split(",")]

            # This is the order in which different fengines should be captured by the capture code
            # This is also then the order of the weights in the delay buffer for each beam
            ordered_feng_ids = sorted(feng_ids) + sorted(list((set(beam_feng_ids) - set(feng_ids))))

            # Create array to hold delay models
            self._delays = np.rec.recarray(nbeams * len(beam_feng_ids), dtype=[
                ("beam_id","uint32"),("feng_id","uint32"),
                ("delay_rate","float32"),("delay_offset","float32")])
            as_bytes = self._delays.tobytes()

            # Create shared memory buffer that will be accessed from beamformer
            self._shared_delay_buffer = posix_ipc.SharedMemory(
                "{}_delay_buffer".format(configuration_id),
                flags=sysv_ipc.IPC_CREX,
                size=len(as_bytes))

            # Compile beamformer
            # TBD

            # Create input DADA buffer
            # TBD

            # Create coherent beam output DADA buffer
            # TBD

            # Create incoherent beam output DADA buffer
            # TBD

            # Create SPEAD transmitter for coherent beams
            # TBD

            # Create SPEAD transmitter for incoherent beam
            # TBD

            # Start beamformer instance
            # TBD

            # Start SPEAD receiver
            # TBD



        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_deconfigure(self, req):
        """
        @brief      Deconfigure the FBFUSE instance.

        @note       Deconfigure the FBFUSE instance. If FBFUSE uses katportalclient to get information
                    from CAM, then it should disconnect at this time.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !deconfigure ok | (fail [error description]) ]]]
        """


        self._shared_delay_buffer.remove()


        return ("ok",)

    @request(Str())
    @return_reply()
    def request_capture_init(self, req):
        """
        @brief      Prepare FBFUSE ingest process for data capture.

        @note       A successful return value indicates that FBFUSE is ready for data capture and
                    has sufficient resources available. An error will indicate that FBFUSE is not
                    in a position to accept data

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being told to start capture.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !capture-init ok | (fail [error description]) ]]]
        """
        return ("ok",)

    @request(Str())
    @return_reply()
    def request_capture_done(self, req):
        """
        @brief      Terminate the FBFUSE ingest process for the particular FBFUSE instance

        @note       This writes out any remaining metadata, closes all files, terminates any remaining processes and
                    frees resources for the next data capture.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being told to stop capture.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !capture-done ok | (fail [error description]) ]]]
        """
        return ("ok",)


@tornado.gen.coroutine
def on_shutdown(ioloop, server):
    log.info("Shutting down server")
    yield server.stop()
    ioloop.stop()

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-H', '--host', dest='host', type=str,
        help='Host interface to bind to')
    parser.add_option('-p', '--port', dest='port', type=long,
        help='Port number to bind to')
    parser.add_option('', '--log_level',dest='log_level',type=str,
        help='Port number of status server instance',default="INFO")
    parser.add_option('', '--dummy',action="store_true", dest='dummy',
        help='Set status server to dummy')
    parser.add_option('-n', '--nodes',dest='nodes', type=str, default=None,
        help='Path to file containing list of available nodes')
    (opts, args) = parser.parse_args()
    FORMAT = "[ %(levelname)s - %(asctime)s - %(filename)s:%(lineno)s] %(message)s"
    logger = logging.getLogger('reynard')
    logging.basicConfig(format=FORMAT)
    logger.setLevel(opts.log_level.upper())
    ioloop = tornado.ioloop.IOLoop.current()
    log.info("Starting FbfControlServer instance")

    if opts.nodes is not None:
        with open(opts.nodes) as f:
            nodes = f.read()
    else:
        nodes = test_nodes

    server = FbfControlServer(opts.host, opts.port, nodes, dummy=opts.dummy)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))

    def start_and_display():
        server.start()
        log.info("Listening at {0}, Ctrl-C to terminate server".format(server.bind_address))
    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    main()
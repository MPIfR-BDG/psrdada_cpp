import logging
import json
import tornado
import signal
import posix_ipc
import ctypes
from threading import Lock
from subprocess import Popen, PIPE, check_call
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
        """
        @brief    Controls shared memory delay buffers that are accessed by one or more
                  beamformer instances.

        @params   delay_engine       A KATCPResourceClient connected to an FBFUSE delay engine server
        @params   nbeams             The number of coherent beams to expect from the delay engine
        @params   nantennas          The number of antennas to expect from the delay engine
        @params   configuration_id   A unique configuration ID that provides a prefix to the names of posix
                                     shared memory buffers created by this instance.
        @params   nreaders           The number of posix shared memory readers that will access the memory
                                     buffers that are managed by this instance.
        """
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
        log.info("Creating mutex semaphore, key='{}'".format(self.mutex_semaphore_key))
        self._mutex_semaphore = posix_ipc.Semaphore(
            self.mutex_semaphore_key,
            flags=posix_ipc.O_CREX,
            initial_value=self._nreaders)

        # This semaphore is used to notify beamformer instances of a change to the
        # delay models. Upon any change its value is simply incremented by one.
        log.info("Creating counting semaphore, key='{}'".format(self.counting_semaphore_key))
        self._counting_semaphore = posix_ipc.Semaphore(self.counting_semaphore_key,
            flags=posix_ipc.O_CREX,
            initial_value=0)

        # This is the share memory buffer that contains the delay models for the
        log.info("Creating shared memory, key='{}'".format(self.shared_buffer_key))
        self._shared_buffer = posix_ipc.SharedMemory(
            self.shared_buffer_key,
            flags=posix_ipc.O_CREX,
            size=len(as_bytes))

        log.info("Regestering delay model update callback")
        self._delay_engine.sensor.delay_model.set_sampling_strategy('event')
        self._delay_engine.sensor.delay_model.register_listener(self._update)

    def __del__(self):
        log.info("Deregestering delay model update callback")
        self._delay_engine.sensor.delay_model.unregister_listener(self._update)
        log.info("Removing shared memory, key='{}'".format(self.shared_buffer_key))
        self._shared_buffer.remove()
        log.info("Removing counting semaphore, key='{}'".format(self.counting_semaphore_key))
        self._counting_semaphore.remove()
        log.info("Removing mutex semaphore, key='{}'".format(self.mutex_semaphore_key))
        self._mutex_semaphore.remove()

    def _update(self, rt, t, status, value):
        # This is a sensor callback to be triggered on
        # the change of the delay models

        log.info("Delay model update triggered - rt:{}, t:{}, status:{}".format(rt,t,status))

        try:
            model = json.loads(value)
        except Exception as error:
            log.exception("Failed to parse delay model JSON: {}".format(value))

        beams = model["beams"]
        antennas = model["antennas"]
        delays = model["model"]
        for ii,beam in enumerate(beams):
            for jj,antenna in enumerate(antennas):
                rate = float(delays[ii,jj,0])
                offset = float(delays[ii,jj,1])
                self._delays_array[self._beam_antenna_map[(beam,antenna)]] = (rate, offset)

        # Acquire the semaphore for each possible reader
        log.debug("Acquiring semaphore for each reader")
        for ii in range(self._nreaders):
            self._mutex_semaphore.acquire()

        # Update the schared memory with the newly acquired model
        log.debug("Writing delay model to shared memory buffer")
        self._shared_buffer.write(self._delays_array.tobytes())

        # Increment the counting semaphore to notify the readers
        # that a new model is available
        log.debug("Incrementing counting semaphore")
        self._counting_semaphore.release()

        # Release the semaphore for each reader
        log.debug("Releasing semaphore for each reader")
        for ii in range(self._nreaders):
            self._mutex_semaphore.release()


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

        @note     The following sensors are made available on top of default sensors
                  implemented in AsynDeviceServer and its base classes.

                  device-status:      Reports the health status of the FBFUSE and associated devices:
                                      Among other things report HW failure, SW failure and observation failure.
        """
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of FbfControlServer instance",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)



    # need to pass the delay engine parameters here

    @request(Str(), Str(), Int(), Int(), Int())
    @return_reply()
    def request_configure(self, req, configuration_id, mcast_groups_json, feng_ids_csv,
        beam_feng_ids_csv, nbeams, nchannels, beams_to_mcast_map, antennas_to_feng_ids_map):
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
            ordered_feng_ids = sorted(beam_feng_ids) + sorted(list((set(feng_ids) - set(beam_feng_ids))))

            # Compile beamformer
            # TBD

            # Need to come up with a good way to allocate keys for dada buffers

            # Create input DADA buffer
            self._dada_input_key = 0xdada
            log.info("Creating dada buffer for input with key '{}'".format(self._dada_input_key))
            check_call(["dada_db","-k",self._dada_input_key,"-n","64","-l","-p"])

            # Create coherent beam output DADA buffer
            self._dada_coh_output_key = 0xcaca
            log.info("Creating dada buffer for coherent beam output with key '{}'".format(self._dada_coh_output_key))
            check_call(["dada_db","-k",self._dada_coh_output_key,"-n","64","-l","-p"])

            # Create incoherent beam output DADA buffer
            self._dada_incoh_output_key = 0xbaba
            log.info("Creating dada buffer for incoherent beam output with key '{}'".format(self._dada_incoh_output_key))
            check_call(["dada_db","-k",self._dada_incoh_output_key,"-n","64","-l","-p"])

            # Create SPEAD transmitter for coherent beams
            # Call to MKSEND

            # Create SPEAD transmitter for incoherent beam
            # Call to MKSEND

            # Start DelayBufferController instance
            # Here we are going to make the assumption that the server and processing all run in
            # one docker container that will be preallocated with the right CPU set, GPUs, memory
            # etc. This means that the configurations need to be unique by NUMA node... [Note: no
            # they don't, we can use the container IPC channel which isolates the IPC namespaces.]
            self._delay_buffer_controller = DelayBufferController(self._delay_engine, nbeams,
                len(beam_feng_ids), configuration_id, 1)

            # Start beamformer instance
            # TBD

            # Define MKRECV configuration file

            # SPEAD receiver does not get started until a capture init call

            req.reply("ok",)

        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_deconfigure(self, req):
        """
        @brief      Deconfigure the FBFUSE instance.

        @note       Deconfigure the FBFUSE instance. If FBFUSE uses katportalclient to get information
                    from CAM, then it should disconnect at this time.

        @param      req               A katcp request object

        @return     katcp reply object [[[ !deconfigure ok | (fail [error description]) ]]]
        """

        # Need to make sure everything is stopped
        # Call self.stop?

        # Need to delete all allocated DADA buffers:

        @tornado.gen.coroutine
        def deconfigure():
            log.info("Destroying dada buffer for input with key '{}'".format(self._dada_input_key))
            check_call(["dada_db","-k",self._dada_input_key,"-d"])
            log.info("Destroying dada buffer for coherent beam output with key '{}'".format(self._dada_coh_output_key))
            check_call(["dada_db","-k",self._dada_coh_output_key,"-n","64","-l","-p"])
            log.info("Destroying dada buffer for incoherent beam output with key '{}'".format(self._dada_incoh_output_key))
            check_call(["dada_db","-k",self._dada_coh_output_key,"-n","64","-l","-p"])
            log.info("Destroying delay buffer controller")
            del self._delay_buffer_controller
            self._delay_buffer_controller = None
            req.reply("ok",)

        self.ioloop.add_callback(deconfigure)
        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_capture_init(self, req):
        """
        @brief      Prepare FBFUSE ingest process for data capture.

        @note       A successful return value indicates that FBFUSE is ready for data capture and
                    has sufficient resources available. An error will indicate that FBFUSE is not
                    in a position to accept data

        @param      req               A katcp request object


        @return     katcp reply object [[[ !capture-init ok | (fail [error description]) ]]]
        """

        # Here we start MKRECV running into the input dada buffer
        self._mkrecv_ingest_proc = Popen(["mkrecv","--config",self._mkrecv_config_filename], stdout=PIPE, stderr=PIPE)
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

        @tornado.gen.coroutine
        def stop_mkrecv_capture():
            #send SIGTERM to MKRECV
            log.info("Sending SIGTERM to MKRECV process")
            self._mkrecv_ingest_proc.terminate()
            self._mkrecv_timeout = 10.0
            log.info("Waiting {} seconds for MKRECV to terminate...".format(self._mkrecv_timeout))
            now = time.time()
            while time.time()-now < self._mkrecv_timeout:
                retval = self._mkrecv_ingest_proc.poll()
                if retval is not None:
                    log.info("MKRECV returned a return value of {}".format(retval))
                    break
                else:
                    time.sleep(0.5)
            else:
                log.warning("MKRECV failed to terminate in alloted time")
                log.info("Killing MKRECV process")
                self._mkrecv_ingest_proc.kill()
            req.reply("ok",)
        self.ioloop.add_callback(self.stop_mkrecv_capture)
        raise AsyncReply


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
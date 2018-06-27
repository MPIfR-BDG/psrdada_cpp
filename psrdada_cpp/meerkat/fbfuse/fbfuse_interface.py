import logging
import json
import tornado
import signal
from threading import Lock
from optparse import OptionParser
from katcp import Sensor, AsyncDeviceServer
from katcp.kattypes import request, return_reply, Int, Str, Discrete

log = logging.getLogger("reynard.fbfuse_interface")

lock = Lock()

def is_power_of_two(n):
    """
    @brief  Test if number is a power of two

    @return True|False
    """
    return n != 0 and ((n & (n - 1)) == 0)

def next_power_of_two(n):
    """
    @brief  Round a number up to the next power of two
    """
    return 2**(n-1).bit_length()

class FbfControlServerWrapper(object):
    def __init__(self, hostname, port):
        self._client = KATCPClientResource(dict(
            name="control-server-client",
            addresss=(hostname, port),
            controlled=True))
        self.hostname = hostname
        self.port = port
        self.priority = 0
        self._started = False

    def start(self):
        self._client.start()
        self._started =True

    def __repr__(self):
        return "FbfControlServer({}:{})".format(self.hostname,self.port)

    def __hash__(self):
        return hash((self.hostname,self.port))

    def __del__(self):
        if self._started:
            try:
                self._client.stop()
            except Exception as error:
                log.exception(str(error))


class FbfControlServerManager(object):
    """Wrapper class for managing server
    allocation and deallocation to subarray/products
    """
    def __init__(self):
        """
        @brief   Construct a new instance

        @param   servers    An iterable container of FbfControlServerWrapper objects
        """
        self._servers = set()
        self._allocated = set()

    def add(self, hostname, port):
        server = FbfControlServerWrapper(hostname,port)
        server.start()
        self._servers.add(server)

    def remove(self, hostname, port):
        self._servers.remove(FbfControlServerWrapper(hostname,port))
        self._allocated.remove(FbfControlServerWrapper(hostname,port))

    def allocate(self, count):
        """
        @brief    Allocate a number of servers from the pool.

        @note     Free servers will be allocated by priority order
                  with 0 being highest priority

        @return   A list of FbfControlServerWrapper objects
        """
        with lock:
            log.debug("Request to allocate {} servers".format(count))
            available_servers = list(self._servers.difference(self._allocated))
            log.debug("{} servers available".format(len(available_servers)))
            available_servers.sort(key=lambda server: server.priority, reverse=True)
            if len(available_servers) < count:
                raise NodeUnavailable("Cannot allocate {0} servers, only {1} available".format(
                    count, len(available_servers)))
            allocated_servers = []
            for _ in range(count):
                server = available_servers.pop()
                log.debug("Allocating server: {}".format(server))
                allocated_servers.append(server)
                self._allocated.add(server)
            return allocated_servers

    def deallocate(self, servers):
        """
        @brief    Deallocate servers and return the to the pool.

        @param    A list of Node objects
        """
        for server in servers:
            log.debug("Deallocating server: {}".format(server))
            self._allocated.remove(server)

    def reset(self):
        """
        @brief   Deallocate all servers
        """
        self._allocated = set()

    def available(self):
        """
        @brief   Return list of available servers
        """
        return list(self._servers.difference(self._allocated))

    def used(self):
        """
        @brief   Return list of allocated servers
        """
        return list(self._allocated)


class FbfMasterController(AsyncDeviceServer):
    """The master pulsar backend control server for
    the Effeelsberg radio telescope.
    """
    VERSION_INFO = ("reynard-fbf-api", 0, 1)
    BUILD_INFO = ("reynard-fbf-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port, dummy=True):
        """
        @brief       Construct new FbfMasterController instance

        @params  ip       The IP address on which the server should listen
        @params  port     The port that the server should bind to
        @params  dummy    Specifies if the instance is running in a dummy mode

        @note   In dummy mode, the controller will act as a mock interface only, sending no requests to nodes.
                A valid node pool must still be provided to the instance, but this may point to non-existent nodes.

        """
        super(FbfMasterController, self).__init__(ip,port)
        self._products = {}
        self._dummy = dummy
        self._server_pool = FbfControlServerManager()

    def start(self):
        """Start FbfMasterController server"""
        super(FbfMasterController,self).start()

    def setup_sensors(self):
        """
        @brief    Set up monitoring sensors.

        @note     The following sensors are made available on top of defaul sensors
                  implemented in AsynDeviceServer and its base classes.

                  device-status:      Reports the health status of the FBFUSE and associated devices:
                                      Among other things report HW failure, SW failure and observation failure.

                  local-time-synced:  Indicates whether the local time of FBFUSE servers
                                      is synchronised to the master time reference (use NTP).
                                      This sensor is aggregated from all nodes that are part
                                      of FBF and will return "not sync'd" if any nodes are
                                      unsyncronised.

        """
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of FBFUSE",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)

        self._local_time_synced = Sensor.boolean(
            "local-time-synced",
            description="Indicates FBF is NTP syncronised.",
            default=True,
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._local_time_synced)

    @request(Str(), Str())
    @return_reply()
    def request_register_control_server(self, req, hostname, port):
        """
        @brief   Register an FbfControlServer instance

        @detail  Register an FbfControlServer instance that can be used for FBFUSE
                 computation. FBFUSE has no preference for the order in which control
                 servers are allocated to a subarray. An FbfControlServer wraps an atomic
                 unit of compute comprised of one CPU, one GPU and one NIC (i.e. one NUMA
                 node on an FBFUSE compute server).
        """
        self._server_pool.add(hostname, port)
        return ("ok",)

    @request(Str(), Str())
    @return_reply()
    def request_deregister_control_server(self, req, hostname, port):
        """
        @brief   Deregister an FbfControlServer instance

        @detail  The graceful way of removing a server from rotation. If the server is
                 currently actively processing a warning will be raised.
        """
        self._server_pool.remove(hostname, port)
        return ("ok",)

    @request(Str(), Str())
    @return_reply(Int())
    def request_list_control_servers(self, req):
        """
        @brief   List all control servers and provide minimal metadata
        """
        for server in self._server_pool.used():
            req.inform("{} allocated".format(server))
        for server in self._server_pool.available():
            req.inform("{} free".format(server))
        return ("ok", len(self._server_pool.used()) + len(self._server_pool.available()))



    @request(Str(), Str(), Int(), Str(), Str())
    @return_reply()
    def request_configure(self, req, product_id, antennas_csv, n_channels, streams_json, proxy_name):
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
                                         {'cbf.antenna_channelised_voltage':
                                             {'i0.antenna-channelised-voltage':'spead://239.2.1.150+15:7148'},
                                          ...}
                                      @endcode

                                      If using katportalclient to get information from CAM, then reconnect and re-subscribe to all sensors
                                      of interest at this time.

        @param      proxy_name        The CAM name for the instance of the FBFUSE data proxy that is being configured.
                                      For example, "FBFUSE_3".  This can be used to query sensors on the correct proxy,
                                      in the event that there are multiple instances in the same subarray.

        @note       A configure call will result in the generation of a new subarray instance in FBFUSE that will be added to the clients list.

        @return     katcp reply object [[[ !configure ok | (fail [error description]) ]]]
        """
        # Test if product_id already exists
        if product_id in self._products:
            return ("fail", "FBF already has a configured product with ID: {}".format(product_id))
        # Determine number of nodes required based on number of antennas in subarray
        # Note this is a poor way of handling this that may be updated later. In theory
        # there is a throughput measure as a function of bandwidth, polarisations and number
        # of antennas that allows one to determine the number of nodes to run. Currently we
        # just assume one antennas worth of data per NIC on our servers, so two antennas per
        # node.
        antennas = antennas_csv.split(",")
        nantennas = len(antennas)
        if not is_power_of_two(nantennas):
            log.warning("Number of antennas was not a power of two. Rounding up to next power of two for resource allocation.")
        required_nodes = next_power_of_two(nantennas)//2
        try:
            nodes = self._node_pool.allocate(required_nodes)
        except Exception as error:
            return ("fail", str(error))
        streams = json.loads(streams_json)
        product = FbfProductController(product_id, antennas, n_channels, streams, proxy_name, nodes)
        try:
            product.start()
            product.configure()
        except Exception as error:
            self._node_pool.deallocate(nodes)
            return ("fail","Error on product configure: {}".format(str(error)))
        #finally if everything is configured successfully we add the product
        #the dictionary of configured products
        self._products[product_id] = product
        return ("ok",)

    @request(Str())
    @return_reply()
    def request_deconfigure(self, req, product_id):
        """
        @brief      Deconfigure the FBFUSE instance.

        @note       Deconfigure the FBFUSE instance. If FBFUSE uses katportalclient to get information
                    from CAM, then it should disconnect at this time.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !deconfigure ok | (fail [error description]) ]]]
        """
        # Test if product exists
        if product_id not in self._products:
            return ("fail", "No product configured with ID: {}".format(product_id))
        product = self._products[product_id]
        try:
            product.deconfigure()
        except Exception as error:
            return ("fail", str(error))
        self._node_pool.deallocate(product.nodes)
        del self._products[product_id]
        return ("ok",)


    @request(Str(), Int(), Str(), Int(), Int())
    @return_reply()
    def request_configure_coherent_beams(self, req, product_id, nbeams, antennas, fscrunch, tscrunch)
        """
        @brief      Request that FBFUSE configure parameters for coherent beams
        """
        if product_id not in self._products:
            return ("fail", "No product configured with ID: {}".format(product_id))
        product = self._products[product_id]
        product.configure_coherent_beams(nbeams, antennas, fscrunch, tscrunch)
        return ("ok",)

    @request(Str(), Str(), Int(), Int())
    @return_reply()
    def request_configure_incoherent_beam(self, req, product_id, antennas, fscrunch, tscrunch)
        """
        @brief      Request that FBFUSE sets the parameters for the incoherent beam
        """
        if product_id not in self._products:
            return ("fail", "No product configured with ID: {}".format(product_id))
        product = self._products[product_id]
        product.configure_incoherent_beam(antennas, fscrunch, tscrunch)
        return ("ok",)

    @request(Str())
    @return_reply()
    def request_start_beams(self, req, product_id):
        """
        @brief      Request that FBFUSE start beams streaming
        """
        if product_id not in self._products:
            return ("fail", "No product configured with ID: {}".format(product_id))
        product = self._products[product_id]

        @tornado.gen.coroutine
        def start():
            product.start_beams()
            req.reply("ok",)

        self.ioloop.add_callback(start)
        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_capture_init(self, req, product_id):
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
        if product_id not in self._products:
            return ("fail", "No product configured with ID: {}".format(product_id))
        product = self._products[product_id]
        try:
            product.capture_init()
        except Exception as error:
            return ("fail",str(error))
        else:
            return ("ok",)

    @request(Str())
    @return_reply()
    def request_capture_done(self, req, product_id):
        """
        @brief      Terminate the FBFUSE ingest process for the particular FBFUSE instance

        @note       This writes out any remaining metadata, closes all files, terminates any remaining processes and
                    frees resources for the next data capture.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being told to stop capture.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !capture-done ok | (fail [error description]) ]]]
        """
        if product_id not in self._products:
            return ("fail", "No product configured with ID: {}".format(product_id))
        product = self._products[product_id]
        try:
            product.capture_done()
        except Exception as error:
            return ("fail",str(error))
        else:
            return ("ok",)

    @request()
    @return_reply(Int())
    def request_products_list(self, req):
        """
        @brief      List all currently registered products and their states

        @param      req               A katcp request object

        @note       The details of each product are provided via an #inform
                    as a JSON string containing information on the product state.
                    For example:

                    @code
                    {'array_1_bc856M4k':{'status':'configured','nodes':['fbf00','fbf01']}}
                    @endcode

        @return     katcp reply object [[[ !capture-done ok | (fail [error description]) <number of configured products> ]]],
        """
        for product_id,product in self._products.items():
            info = {}
            info[product_id] = {}
            info[product_id]['status'] = "capturing" if product.capturing else "configured"
            info[product_id]['nodes'] = [i.hostname for i in product.nodes]
            as_json = json.dumps(info)
            req.inform(as_json)
        return ("ok",len(self._products))


class Beam(object):
    def __init__(self, idx, ra=0, dec=0, source_name="unknown"):
        self.idx = idx
        self.source_name = source_name
        self.ra = ra
        self.dec = dec


class Tiling(object):
    def __init__(self, ra, dec, source_name, reference_frequency, overlap):
        self._beams = []
        self.ra = ra
        self.dec = dec
        self.source_name = source_name
        self.reference_frequency = reference_frequency
        self.overlap = overlap

    def add_beam(self, beam):
        self._beams.append(beam)

    def generate(self, epoch, antennas):
        print "Updating RA and Decs of beams"


class DynamicTiling(Tiling):
    def __init__(self, ra, dec, source_name, reference_frequency, overlap, tolerance):
        super(DynamicTiling, self).__init__(ra, dec, source_name, reference_frequency, overlap)
        self.tolerance = tolerance
        self._update_cycle = 30.0
        self._update_callback = None

    def start_update_loop(self, ioloop):
        #self._update_callback = PeriodicCallback(self.generate, )
        pass


class BeamManager(object):
    def __init__(self, nbeams, antennas):
        self._nbeams = nbeams
        self._antennas = antennas
        self.reset()

    @property
    def nbeams(self):
        return self._nbeams

    @property
    def antennas(self):
        return self._antennas

    def reset(self):
        self._free_beams = [Beam("cfbf%05d"%(i)) for i in range(self._nbeams)]
        self._allocated_beams = []
        self._tilings = []
        self._dynamic_tilings = []

    def add_beam(self, ra, dec, source_name):
        beam = self._free_beams.pop(0)
        beam.ra = ra
        beam.dec = dec
        beam.source_name = source_name
        self._allocated_beams.append(beam)
        return beam

    def __make_tiling(self, nbeams, tiling_type, *args):
        if len(self._free_beams) < nbeams:
            raise Exception("More beams requested than are available.")
        tiling = tiling_type(*args)
        for _ in range(nbeams):
            beam = self._free_beams.pop(0)
            tiling.add_beam(beam)
            self._allocated_beams.append(beam)
        return tiling

    def add_tiling(self, ra, dec, source_name, nbeams, reference_frequency, overlap):
        tiling = self.__make_tiling(nbeams, Tiling, ra, dec, source_name,
            reference_frequency, overlap)
        self._tilings.append(tiling)
        return tiling

    def add_dynamic_tiling(self, ra, dec, source_name, nbeams, reference_frequency, overlap, tolerance):
        tiling = self.__make_tiling(nbeams, DynamicTiling, ra, dec, source_name,
            reference_frequency, overlap, tolerance)
        self._dynamic_tilings.append(tiling)
        return tiling

    def get_beams(self):
        return self._allocated_beams + self._free_beams


class DelayEngine(AsyncDeviceServer):
    """The master pulsar backend control server for
    the Effeelsberg radio telescope.
    """
    VERSION_INFO = ("delay-engine-api", 0, 1)
    BUILD_INFO = ("delay-engine-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port, beam_manager):
        self._beam_manager = beam_manager
        super(DelayEngine, self).__init__(ip,port)

    def setup_sensors(self):
        """
        @brief    Set up monitoring sensors.

        Sensor list:
        - beams
        - antennas
        - delays
        - update-rate
        """
        self._update_rate_sensor = Sensor.float(
            "update-rate",
            description="The delay update rate",
            default=2.0,
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._update_rate_sensor)
        self._beams_sensor = Sensor.string(
            "beams",
            description="JSON breakdown of the beams associated with this delay engine",
            default="{}",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._beams_sensor)
        self._nbeams_sensor = Sensor.int(
            "nbeams",
            description="Number of beams that this delay engine handles",
            default=0,
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._nbeams_sensor)
        self._antennas_sensor = Sensor.string(
            "antennas",
            description="JSON breakdown of the antennas (in KATPOINT format) associated with this delay engine",
            default="{}",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._antennas_sensor)
        self._delays_sensor = Sensor.string(
            "delays",
            description="JSON object containing delays for each beam for each antenna at the current epoch",
            default="",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._delays_sensor)

    @request(Float())
    @return_reply()
    def request_set_update_rate(self, req, rate):
        """
        @brief    Set the update rate for delay calculations
        """
        self._update_rate_sensor.set_value(rate)
        # This should make a change to the beam manager object
        return ("ok",)


class BeamSetConfig(object):
    def __init__(self, nbeams, fscrunch, tscrunch, antennas):
        self.nbeams = nbeams
        self.fscrunch = fscrunch
        self.tscrunch = tscrunch
        self.antennas = antennas


class FbfProductController(object):
    """
    Wrapper class for an FBFUSE product. Objects of this type create a UBI server instance and
    allocate nodes to this. The intention here is to have a class that translates information
    specific to MeerKAT into a general configuration and pipeline deployment tool as is done
    for Effelsberg.
    """
    def __init__(self, product_id, antennas, n_channels, streams, proxy_name, servers):
        """
        @brief      Construct new instance

        @param      product_id        The name of the product

        @param      antennas_csv      A list of antenna names

        @param      n_channels        The integer number of frequency channels provided by the CBF.

        @param      streams           A dictionary containing config keys and values describing the streams.
        """
        self._product_id = product_id
        self._antennas = antennas
        self._n_channels = n_channels
        self._streams = streams
        self._proxy_name = proxy_name
        self._servers = servers
        self._capturing = False
        self._beam_manager = None
        self._delay_engine = None
        self._coherent_beam_config = BeamSetConfig(400, 1, 1, antennas)
        self._incoherent_beam_config = BeamSetConfig(1, 1, 1, antennas)

    @property
    def servers(self):
        return self._servers

    @property
    def capturing(self):
        return self._capturing

    def configure_coherent_beams(self, nbeams, antennas, fscrunch, tscrunch):
        if self.capturing:
            raise Exception("Configuration calls must be made before start_beams is called")
        self._coherent_beam_config = BeamSetConfig(nbeams, fscrunch, tscrunch, antennas)

    def configure_incoherent_beam(self, antennas, fscrunch, tscrunch):
        if self.capturing:
            raise Exception("Configuration calls must be made before start_beams is called")
        self._incoherent_beam_config = BeamSetConfig(1, fscrunch, tscrunch, antennas)

    def start_beams(self):
        """
        @brief      start_beams
        """
        if self.capturing:
            raise Exception("Beam streaming has already been started")

        self._beam_manager = BeamManager(self._coherent_beam_config.nbeams, self._coherent_beam_config.antennas)
        self._delay_engine = DelayEngine("127.0.0.1", 0, self._beam_manager)
        self._delay_engine.start()

        for server in self._servers:
            # each server will take 4 consequtive multicast groups
            # Configure call should probably start the streaming as well
            #yield server.req.configure(...)
            pass

        # set up delay engine
        # compile kernels
        # start streaming
        self._capturing = True
        pass

    def stop_beams(self):
        if not self.capturing:
            return
        for server in self._servers:
            #yield server.req.deconfigure()
            pass

    def add_beam(self, ra, dec, source_name):
        if not self.capturing:
            raise Exception("Beam configurations should be specified after a call to start_beams")
        self._beam_manager.add_beam(ra, dec, source_name)

    def add_tiling(self, ra, dec, source_name, number_of_beams, reference_frequency, overlap, epoch):
        if not self.capturing:
            raise Exception("Tiling configurations should be specified after a call to start_beams")
        tiling = self._beam_manager.add_tiling(ra, dec, source_name, number_of_beams, reference_frequency, overlap)
        tiling.generate(epoch)

    def add_dynamic_tiling(self, ra, dec, source_name, number_of_beams, reference_frequency, overlap, tolerance):
        if not self.capturing:
            raise Exception("Tiling configurations should be specified after a call to start_beams")
        tiling = self._beam_manager.add_tiling(ra, dec, source_name, number_of_beams, reference_frequency, overlap, tolerance)
        tiling.start_update_loop()

    def reset_beams(self):
        if not self.capturing:
            raise Exception("Beam reset can only be performed after a call to start_beams")
        self._beam_manager.reset()


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
    log.info("Starting FbfMasterController instance")

    if opts.nodes is not None:
        with open(opts.nodes) as f:
            nodes = f.read()
    else:
        nodes = test_nodes

    server = FbfMasterController(opts.host, opts.port, nodes, dummy=opts.dummy)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))

    def start_and_display():
        server.start()
        log.info("Listening at {0}, Ctrl-C to terminate server".format(server.bind_address))
    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    main()




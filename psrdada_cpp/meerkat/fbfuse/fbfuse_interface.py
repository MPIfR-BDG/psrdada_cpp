import logging
import json
import tornado
import signal
import time
import copy
from threading import Lock
from optparse import OptionParser
from katcp import Sensor, Message, AsyncDeviceServer, KATCPClientResource, AsyncReply
from katcp.kattypes import request, return_reply, Int, Str, Discrete, Float
from katportalclient import KATPortalClient
from katpoint import Antenna, Target

log = logging.getLogger("reynard.fbfuse_interface")

lock = Lock()

###################
# Utility functions
###################

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


###################
# Custom exceptions
###################

class ServerAllocationError(Exception):
    pass

class ProductLookupError(Exception):
    pass

class ProductExistsError(Exception):
    pass

class KatportalClientWrapper(object):
    def __init__(self, host):
        self._client = KATPortalClient('http://{host}/api/client'.format(host=host),
            on_update_callback=None, logger=logging.getLogger('katcp'))

    @tornado.gen.coroutine
    def get_observer_string(self, antenna):
        sensor_name = "{}_observer".format(antenna)
        now = time.time()
        start = now - 3600.0
        history = yield self._client.sensor_history(
            sensor_name, start, now, timeout_sec = 20.0)
        raise tornado.gen.Return(history[0].value)

####################
# Classes for communicating with and wrapping
# the functionality of the processing servers
# on each NUMA node.
####################

class FbfWorkerWrapper(object):
    """Wrapper around a client to an FbfWorkerServer
    instance.
    """
    def __init__(self, hostname, port):
        """
        @brief  Create a new wrapper around a client to a worker server

        @params hostname The hostname for the worker server
        @params port     The port number that the worker server serves on
        """
        log.debug("Building client to FbfWorkerServer at {}:{}".format(hostname, port))
        self._client = KATCPClientResource(dict(
            name="worker-server-client",
            address=(hostname, port),
            controlled=True))
        self.hostname = hostname
        self.port = port
        self.priority = 0 # Currently no priority mechanism is implemented
        self._started = False

    def start(self):
        """
        @brief  Start the client to the worker server
        """
        log.debug("Starting client to FbfWorkerServer at {}:{}".format(self.hostname, self.port))
        self._client.start()
        self._started = True

    def __repr__(self):
        return "<{} for {}:{}>".format(self.__class__, self.hostname, self.port)

    def __hash__(self):
        # This has override is required to allow these wrappers
        # to be used with set() objects. The implication is that
        # the combination of hostname and port is unique for a
        # worker server
        return hash((self.hostname, self.port))

    def __eq__(self, other):
        # Also implemented to help with hashing
        # for sets
        return self.__hash__() == hash(other)

    def __del__(self):
        if self._started:
            try:
                self._client.stop()
            except Exception as error:
                log.exception(str(error))


class FbfWorkerPool(object):
    """Wrapper class for managing server
    allocation and deallocation to subarray/products
    """
    def __init__(self):
        """
        @brief   Construct a new instance
        """
        self._servers = set()
        self._allocated = set()

    def add(self, hostname, port):
        """
        @brief  Add a new FbfWorkerServer to the server pool

        @params hostname The hostname for the worker server
        @params port     The port number that the worker server serves on
        """
        wrapper = FbfWorkerWrapper(hostname,port)
        if not wrapper in self._servers:
            wrapper.start()
            log.debug("Adding {} to server set".format(wrapper))
            self._servers.add(wrapper)

    def remove(self, hostname, port):
        """
        @brief  Add a new FbfWorkerServer to the server pool

        @params hostname The hostname for the worker server
        @params port     The port number that the worker server serves on
        """
        wrapper = FbfWorkerWrapper(hostname,port)
        if wrapper in self._allocated:
            raise Exception("Cannot remove allocated server from pool")
        try:
            self._servers.remove(wrapper)
        except KeyError:
            log.warning("Could not remove server at {}:{} from server pool")

    def allocate(self, count):
        """
        @brief    Allocate a number of servers from the pool.

        @note     Free servers will be allocated by priority order
                  with 0 being highest priority

        @return   A list of FbfWorkerWrapper objects
        """
        with lock:
            log.debug("Request to allocate {} servers".format(count))
            available_servers = list(self._servers.difference(self._allocated))
            log.debug("{} servers available".format(len(available_servers)))
            available_servers.sort(key=lambda server: server.priority, reverse=True)
            if len(available_servers) < count:
                raise ServerAllocationError("Cannot allocate {0} servers, only {1} available".format(
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
        log.debug("Reseting server pool allocations")
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

####################
# The main CAM interface for FBFUSE
####################

class FbfMasterController(AsyncDeviceServer):
    """This is the main KATCP interface for the FBFUSE
    multi-beam beamformer on MeerKAT.

    This interface satisfies the following ICDs:
    CAM-FBFUSE: <link>
    TUSE-FBFUSE: <link>
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
        self._server_pool = FbfWorkerPool()

    def start(self):
        """
        @brief  Start the FbfMasterController server
        """
        super(FbfMasterController,self).start()

    def setup_sensors(self):
        """
        @brief  Set up monitoring sensors.

        @note   The following sensors are made available on top of defaul sensors
                implemented in AsynDeviceServer and its base classes.

                device-status:  Reports the health status of the FBFUSE and associated devices:
                                Among other things report HW failure, SW failure and observation failure.

                local-time-synced:  Indicates whether the local time of FBFUSE servers
                                    is synchronised to the master time reference (use NTP).
                                    This sensor is aggregated from all nodes that are part
                                    of FBF and will return "not sync'd" if any nodes are
                                    unsyncronised.

                products:   The list of product_ids that FBFUSE is currently handling
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

        self._products_sensor = Sensor.string(
            "products",
            description="The names of the currently configured products",
            default="",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._products_sensor)

    def _update_products_sensor(self):
        self._products_sensor.set_value(",".join(self._products.keys()))

    def _get_product(self, product_id):
        if product_id not in self._products:
            raise ProductLookupError("No product configured with ID: {}".format(product_id))
        else:
            return self._products[product_id]

    @request(Str(), Str())
    @return_reply()
    def request_register_worker_server(self, req, hostname, port):
        """
        @brief   Register an FbfWorker instance

        @params hostname The hostname for the worker server
        @params port     The port number that the worker server serves on

        @detail  Register an FbfWorker instance that can be used for FBFUSE
                 computation. FBFUSE has no preference for the order in which control
                 servers are allocated to a subarray. An FbfWorker wraps an atomic
                 unit of compute comprised of one CPU, one GPU and one NIC (i.e. one NUMA
                 node on an FBFUSE compute server).
        """
        self._server_pool.add(hostname, port)
        return ("ok",)

    @request(Str(), Str())
    @return_reply()
    def request_deregister_worker_server(self, req, hostname, port):
        """
        @brief   Deregister an FbfWorker instance

        @params hostname The hostname for the worker server
        @params port     The port number that the worker server serves on

        @detail  The graceful way of removing a server from rotation. If the server is
                 currently actively processing an exception will be raised.
        """
        self._server_pool.remove(hostname, port)
        return ("ok",)

    @request()
    @return_reply(Int())
    def request_worker_server_list(self, req):
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
                                      to which the data products belongs (e.g. m007,m008,m009).

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
            raise ProductExistsError("FBF already has a configured product with ID: {}".format(product_id))
        # Determine number of nodes required based on number of antennas in subarray
        # Note this is a poor way of handling this that may be updated later. In theory
        # there is a throughput measure as a function of bandwidth, polarisations and number
        # of antennas that allows one to determine the number of nodes to run. Currently we
        # just assume one antennas worth of data per NIC on our servers, so two antennas per
        # node.
        @tornado.gen.coroutine
        def configure():
            antennas = antennas_csv.split(",")
            nantennas = len(antennas)
            if not is_power_of_two(nantennas):
                log.warning("Number of antennas was not a power of two. Rounding up to next power of two for resource allocation.")
            required_servers = next_power_of_two(nantennas)//2
            # This may be removed in future.
            # Currently if _dummy is set no actual server allocation will be requested.
            if not self._dummy:
                servers = self._server_pool.allocate(required_servers)
            else:
                servers = []
            streams = json.loads(streams_json)
            kpc = KatportalClientWrapper("portal.mkat.karoo.kat.ac.za")
            futures, observers = [],[]
            antennas = antennas_csv.split(",")
            for antenna in antennas:
                log.debug("Fetching katpoint string for antenna {}".format(antenna))
                futures.append(kpc.get_observer_string(antenna))
            for ii,future in enumerate(futures):
                try:
                    observer = yield future
                except Exception as error:
                    print error
                    req.inform("Error retrieving katpoint string for {}".format(antennas[ii]))
                else:
                    log.debug("Fetched katpoint antenna: {}".format(observer))
                    observers.append(Antenna(observer))
            product = FbfProductController(self, product_id, observers, n_channels, streams, proxy_name, servers)
            self._products[product_id] = product
            self._update_products_sensor()
            req.reply("ok",)
        self.ioloop.add_callback(configure)
        raise AsyncReply

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
        product = self._get_product(product_id)
        try:
            product.stop_beams()
        except Exception as error:
            return ("fail", str(error))
        self._server_pool.deallocate(product.servers)
        del self._products[product_id]
        self._update_products_sensor()
        return ("ok",)


    @request(Str(), Int(), Str(), Int(), Int())
    @return_reply()
    def request_configure_coherent_beams(self, req, product_id, nbeams, antennas_csv, fscrunch, tscrunch):
        """
        @brief      Request that FBFUSE configure parameters for coherent beams

        @note       This call can only be made prior to a call to start-beams for the configured product.
                    This is due to FBFUSE requiring static information up front in order to compile beamformer
                    kernels, allocate the correct size memory buffers and subscribe to the correct number of
                    multicast groups.

        @note       The particular configuration passed at this stage will only be evaluated on a call to start-beams.
                    If the requested configuration is not possible due to hardware and bandwidth limits and error will
                    be raised on the start-beams call.

        @param      req             A katcp request object

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      nbeams          The number of beams that will be produced for the provided product_id

        @param      antennas_csv    A comma separated list of physical antenna names. Only these antennas will be used
                                    when generating coherent beams (e.g. m007,m008,m009). The antennas provided here must
                                    be a subset of the antennas in the current subarray. If not an exception will be
                                    raised.

        @param      fscrunch        The number of frequency channels to integrate over when producing coherent beams.

        @param      tscrunch        The number of time samples to integrate over when producing coherent beams.

        @return     katcp reply object [[[ !configure-coherent-beams ok | (fail [error description]) ]]]
        """
        antennas = antennas_csv.split(",")
        product = self._get_product(product_id)
        product.configure_coherent_beams(nbeams, antennas, fscrunch, tscrunch)
        return ("ok",)

    @request(Str(), Str(), Int(), Int())
    @return_reply()
    def request_configure_incoherent_beam(self, req, product_id, antennas_csv, fscrunch, tscrunch):
        """
        @brief      Request that FBFUSE sets the parameters for the incoherent beam

        @note       The particular configuration passed at this stage will only be evaluated on a call to start-beams.
                    If the requested configuration is not possible due to hardware and bandwidth limits and error will
                    be raised on the start-beams call.

        @note       Currently FBFUSE is only set to produce one incoherent beam per instantiation. This may change in future.

        @param      req             A katcp request object

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      nbeams          The number of beams that will be produced for the provided product_id

        @param      antennas_csv    A comma separated list of physical antenna names. Only these antennas will be used
                                    when generating the incoherent beam (e.g. m007,m008,m009). The antennas provided here must
                                    be a subset of the antennas in the current subarray. If not an exception will be
                                    raised.

        @param      fscrunch        The number of frequency channels to integrate over when producing the incoherent beam.

        @param      tscrunch        The number of time samples to integrate over when producing the incoherent beam.

        @return     katcp reply object [[[ !configure-incoherent-beam ok | (fail [error description]) ]]]
        """
        antennas = antennas_csv.split(",")
        product = self._get_product(product_id)
        product.configure_incoherent_beam(antennas, fscrunch, tscrunch)
        return ("ok",)

    @request(Str())
    @return_reply()
    def request_capture_start(self, req, product_id):
        """
        @brief      Request that FBFUSE start beams streaming

        @detail     Upon this call the provided coherent and incoherent beam configurations will be evaluated
                    to determine if they are physical and can be met with the existing hardware. If the configurations
                    are acceptable then servers allocated to this instance will be triggered to begin production of beams.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !start-beams ok | (fail [error description]) ]]]
        """
        timeout = 20.0
        start = time.time()
        product = self._get_product(product_id)
        @tornado.gen.coroutine
        def start():
            state = product._state_sensor.value()
            if state == FbfProductController.STARTING:
                while time.time() < start + timeout:
                    yield tornado.gen.sleep(0.5)
                    state = product._state_sensor.value()
                    if state == FbfProductController.self.CAPTURING:
                        req.reply("ok",)
                        break
                    elif state == FbfProductController.STARTING:
                        continue
                    else:
                        req.reply("fail", "Expected product state to go from 'starting' -> "
                            "'capturing' but it went 'starting' -> '{}'".format(state))
                else:
                    req.reply("fail", "Exceeded timeout on transition from 'starting' to 'capturing' state")
            elif state == FbfProductController.CAPTURING:
                req.reply("ok",)
            elif state == FbfProductController.STOPPING:
                req.reply("fail", "Product is in the process of stopping")
            elif state == FbfProductController.IDLE:
                product.start_beams()
                # Here we should proxy the relevant sensors from the delay engine
                # for the product controller up to this level
                req.reply("ok",)

        self.ioloop.add_callback(start)
        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_provision_beams(self, req, product_id):
        """
        @brief      Request that FBFUSE asynchronously start beams streaming

        @detail     Upon this call the provided coherent and incoherent beam configurations will be evaluated
                    to determine if they are physical and can be met with the existing hardware. If the configurations
                    are acceptable then servers allocated to this instance will be triggered to begin production of beams.
                    Unlike a call to ?capture-start, ?provision-beams will not wait for completion before returning, instead
                    it will start the process of beamformer resource alloction and compilation. To determine when the process
                    is complete, the user must wait on the value of the product "state" sensor becoming "capturing", e.g.

                    @code
                        client.sensor['{}-state'.format(proxy_name)].wait(
                            lambda reading: reading.value == 'capturing')
                    @endcode

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !start-beams ok | (fail [error description]) ]]]
        """
        product = self._get_product(product_id)
        @tornado.gen.coroutine
        def start():
            product.start_beams()

        state = product._state_sensor.value()
        if state != FbfProductController.IDLE:
            return ("fail", "Can only provision beams on an idle FBF product")
        else:
            self.ioloop.add_callback(start)
            return ("ok",)


    @request(Str())
    @return_reply()
    def request_capture_stop(self, req, product_id):
        """
        @brief      Stop FBFUSE streaming

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".
        """
        product = self._get_product(product_id)
        @tornado.gen.coroutine
        def stop():
            product.stop_beams()
            # Here we should proxy the relevant sensors from the delay engine
            # for the product controller up to this level
            req.reply("ok",)
        self.ioloop.add_callback(stop)
        raise AsyncReply

    @request(Str(), Str())
    @return_reply(Str())
    def request_add_beam(self, req, product_id, target):
        """
        @brief      Configure the parameters of one beam

        @note       This call may only be made AFTER a successful call to start-beams. Before this point no beams are
                    allocated to the instance. If all beams are currently allocated an exception will be raised.

        @param      req             A katcp request object

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      target          A KATPOINT target string

        @return     katcp reply object [[[ !add-beam ok | (fail [error description]) ]]]
        """
        target = Target(target)
        beam = self._get_product(product_id).add_beam(target)
        return ("ok", beam.idx)

    @request(Str(), Str(), Int(), Float(), Float(), Float())
    @return_reply(Str())
    def request_add_tiling(self, req, product_id, target, nbeams, reference_frequency, overlap, epoch):
        """
        @brief      Configure the parameters of a static beam tiling

        @note       This call may only be made AFTER a successful call to start-beams. Before this point no beams are
                    allocated to the instance. If there are not enough free beams to satisfy the request an
                    exception will be raised.

        @note       Beam shapes calculated for tiling are always assumed to be 2D elliptical Gaussians.

        @param      req             A katcp request object

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      target          A KATPOINT target string

        @param      nbeams          The number of beams in this tiling pattern.

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @param      epoch           The desired epoch for the tiling pattern as a unix time. A typical usage would
                                    be to set the epoch to half way into the coming observation in order to minimise
                                    the effect of parallactic angle and array projection changes altering the shape
                                    and position of the beams and thus changing the efficiency of the tiling pattern.


        @return     katcp reply object [[[ !add-tiling ok | (fail [error description]) ]]]
        """
        product = self._get_product(product_id)
        target = Target(target)
        tiling = product.add_tiling(target, nbeams, reference_frequency, overlap, epoch)
        return ("ok", tiling.idxs())

    @request(Str(), Str(), Int(), Float(), Float(), Float())
    @return_reply(Str())
    def request_add_dynamic_tiling(self, req, product_id, target, nbeams, reference_frequency, overlap, tolerance):
        """
        @brief      Configure the parameters of one beam

        @note       This call may only be made AFTER a successful call to start-beams. Before this point no beams are
                    allocated to the instance. If there are not enough free beams to satisfy the request an
                    exception will be raised.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @param      target          A KATPOINT target string

        @param      nbeams          The number of beams in this tiling pattern.

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @param      tolerance       The tolerance criterion for triggering a re-tiling of the beam pattern. The tolerance
                                    is based around the percentage change in tiling efficiency. The tiling efficiency is
                                    calculated based on a combined measure of beam separation and covariance. If this all
                                    sounds nebulous and loose, thats because it is. A value of 0.1 corresponds to a 10%
                                    change in tiling efficency.

        @return     katcp reply object [[[ !add-dynamic-tiling ok | (fail [error description]) ]]]
        """
        product = self._get_product(product_id)
        target = Target(target)
        tiling = product.add_tiling(ra, dec, source_name, nbeams, reference_frequency, overlap, epoch)
        return ("ok", tiling.idxs())

    @request()
    @return_reply(Int())
    def request_product_list(self, req):
        """
        @brief      List all currently registered products and their states

        @param      req               A katcp request object

        @note       The details of each product are provided via an #inform
                    as a JSON string containing information on the product state.

        @return     katcp reply object [[[ !product-list ok | (fail [error description]) <number of configured products> ]]],
        """
        for product_id,product in self._products.items():
            info = {}
            info[product_id] = product.info()
            as_json = json.dumps(info)
            req.inform(as_json)
        return ("ok",len(self._products))

    @request(Str())
    @return_reply()
    def request_set_default_target_configuration(self, req, product_id, target):
        """
        @brief      Set the configuration of FBFUSE from the FBFUSE configuration server

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      target          A KATPOINT target string
        """
        product = self._get_product(product_id)
        target = katpoint.Target(target)
        if not product.capturing:
            raise Exception("Product must be capturing before a target confiugration can be set.")
        product.reset_beams()
        # TBD: Here we connect to some database and request the default configurations
        # For example this may return secondary target in the FoV
        #
        # As a default the current system will put one beam directly on target and
        # the rest of the beams in a static tiling pattern around this target
        now = time.time()
        nbeams = product._beam_manager.nbeams
        product.add_tiling(target, nbeams-1, 1.4e9, 0.5, now)
        product.add_beam(target)
        return ("ok",)

    @request(Str())
    @return_reply()
    def request_set_default_sb_configuration(self, req, product_id, sb_id):
        """
        @brief      Set the configuration of FBFUSE from the FBFUSE configuration server

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      sb_id           The schedule block ID. Decisions of the configuarion of FBFUSE will be made dependent on
                                    the configuration of the current subarray, the primary and secondary science projects
                                    active and the targets expected to be visted during the execution of the schedule block.
        """
        product = self._get_product(product_id)
        if product.capturing:
            raise Exception("Cannot reconfigure a currently capturing instance.")
        product.configure_coherent_beams(400, product._katpoint_antennas, 1, 16)
        product.configure_incoherent_beam(product._katpoint_antennas, 1, 16)
        now = time.time()
        nbeams = product._beam_manager.nbeams
        product.add_tiling(target, nbeams-1, 1.4e9, 0.5, now)
        product.add_beam(target)
        return ("ok",)

####################
# Classes representing the different
# beams and tilings that can be provided
# by FBFUSE
####################

DEFAULT_KATPOINT_TARGET = "unset, radec, 0, 0"

class Beam(object):
    """Wrapper class for a single beam to be produced
    by FBFUSE"""
    def __init__(self, idx, target=DEFAULT_KATPOINT_TARGET):
        """
        @brief   Create a new Beam object

        @params   idx   a unique identifier for this beam.

        @param      target          A KATPOINT target object
        """
        self.idx = idx
        self._target = target
        self._observers = set()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target):
        self._target = new_target
        self.notify()

    def notify(self):
        """
        @brief  Notify all observers of a change to the beam parameters
        """
        for observer in self._observers:
            observer(self)

    def register_observer(self, func):
        """
        @brief   Register an observer to be called on a notify

        @params  func  Any function that takes a Beam object as its only argument
        """
        self._observers.add(func)

    def deregister_observer(self, func):
        """
        @brief   Deregister an observer to be called on a notify

        @params  func  Any function that takes a Beam object as its only argument
        """
        self._observers.remove(func)

    def reset(self):
        """
        @brief   Reset the beam to default parameters
        """
        self._target = Target(DEFAULT_KATPOINT_TARGET)

    def __repr__(self):
        return "{}, {}".format(
            self.idx, self.target.format_katcp())


class Tiling(object):
    """Wrapper class for a collection of beams in a tiling pattern
    """
    def __init__(self, target, reference_frequency, overlap):
        """
        @brief   Create a new tiling object

        @param      target          A KATPOINT target object

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]
        """
        self._beams = []
        self.target = target
        self.reference_frequency = reference_frequency
        self.overlap = overlap

    def add_beam(self, beam):
        """
        @brief   Add a beam to the tiling pattern

        @param   beam   A Beam object
        """
        self._beams.append(beam)

    def generate(self, epoch, antennas):
        """
        @brief   Calculate and update RA and Dec positions of all
                 beams in the tiling object.

        @param      epoch     The epoch of tiling (unix time)

        @param      antennas  The antennas to use when calculating the beam shape.
                              Note these are the antennas in katpoint CSV format.
        """
        print "Updating RA and Decs of beams"

    def __repr__(self):
        return ", ".join([repr(beam) for beam in self._beams])

    def idxs(self):
        return ",".join([beam.idx for beam in self._beams])


class DynamicTiling(Tiling):
    """Subclass of Tiling that provide mechanisms
    for periodically updating the tiling pattern"""
    def __init__(self, target_reference_frequency, overlap, tolerance):
        """
        @brief   Create a new dynamic tiling object

        @param      target          A KATPOINT target object

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @param      tolerance       The tolerance criterion for triggering a re-tiling of the beam pattern. The tolerance
                                    is based around the percentage change in tiling efficiency. The tiling efficiency is
                                    calculated based on a combined measure of beam separation and covariance. If this all
                                    sounds nebulous and loose, thats because it is. A value of 0.1 corresponds to a 10%
                                    change in tiling efficency.
        """
        super(DynamicTiling, self).__init__(target, reference_frequency, overlap)
        self.tolerance = tolerance
        self._update_cycle = 30.0
        self._update_callback = None

    def start_update_loop(self, ioloop):
        #TBD
        pass


class BeamManager(object):
    """Manager class for allocation, deallocation and tracking of
    individual beams, static and dynamic tilings.
    """
    def __init__(self, nbeams, antennas):
        """
        @brief  Create a new beam manager object

        @param  nbeams    The number of beams managed by this object

        @param  antennas  A list of antennas to use for tilings. Note these should
                          be in KATPOINT CSV format.
        """
        self._nbeams = nbeams
        self._antennas = antennas
        self._beams = [Beam("cfbf%05d"%(i)) for i in range(self._nbeams)]
        self.reset()

    @property
    def nbeams(self):
        return self._nbeams

    @property
    def antennas(self):
        return self._antennas

    def reset(self):
        """
        @brief  reset and deallocate all beams and tilings managed by this instance

        @note   All tiling will be lost on this call and must be remade for subsequent observations
        """
        for beam in self._beams:
            beam.reset()
        self._free_beams = copy.copy(self._beams)
        self._allocated_beams = []
        self._tilings = []
        self._dynamic_tilings = []

    def __add_beam(self, target):
        beam = self._free_beams.pop(0)
        beam.target = target
        self._allocated_beams.append(beam)
        return beam

    def add_beam(self, target):
        """
        @brief   Specify the parameters of one managed beam

        @param      target          A KATPOINT target object

        @return     Returns the allocated Beam object
        """
        beam = self.__add_beam(target)
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

    def add_tiling(self, target, nbeams, reference_frequency, overlap):
        """
        @brief   Add a tiling to be managed

        @param      target          A KATPOINT target object

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @returns    The created Tiling object
        """
        tiling = self.__make_tiling(nbeams, Tiling, target,
            reference_frequency, overlap)
        self._tilings.append(tiling)
        return tiling

    def add_dynamic_tiling(self, target, nbeams, reference_frequency, overlap, tolerance):
        """
        @brief   Create a new dynamic tiling object

        @param      target          A KATPOINT target object

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @param      tolerance       The tolerance criterion for triggering a re-tiling of the beam pattern. The tolerance
                                    is based around the percentage change in tiling efficiency. The tiling efficiency is
                                    calculated based on a combined measure of beam separation and covariance. If this all
                                    sounds nebulous and loose, thats because it is. A value of 0.1 corresponds to a 10%
                                    change in tiling efficency.

        @returns    The created DynamicTiling object
        """
        tiling = self.__make_tiling(nbeams, DynamicTiling, target,
            reference_frequency, overlap, tolerance)
        self._dynamic_tilings.append(tiling)
        return tiling

    def get_beams(self):
        """
        @brief  Return all managed beams
        """
        return self._allocated_beams + self._free_beams


####################
# Classes for wrapping an individual
# configuration of FBFUSE (i.e. an FBF product)
####################

class DelayEngine(AsyncDeviceServer):
    """A server for maintining delay models used
    by FbfWorkerServers.
    """
    VERSION_INFO = ("delay-engine-api", 0, 1)
    BUILD_INFO = ("delay-engine-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port, beam_manager):
        """
        @brief  Create a new DelayEngine instance

        @param   ip   The interface that the DelayEngine should serve on

        @param   port The port that the DelayEngine should serve on

        @param   beam_manager  A BeamManager instance that will be used to create delays
        """
        self._beam_manager = beam_manager
        super(DelayEngine, self).__init__(ip,port)

    def setup_sensors(self):
        """
        @brief    Set up monitoring sensors.

        @note     The key sensor here is the delay sensor which is stored in JSON format

                  @code
                  {
                  'antenns':['m007','m008','m009'],
                  'beams':['cfbf00001','cfbf00002'],
                  'model': [[[0,2],[0,5]],[[2,3],[4,4]],[[8,8],[8,8]]]
                  }
                  @endcode

                  Here the delay model is stored as a 3 dimensional array
                  with dimensions of beam, antenna, model (rate,offset) from
                  outer to inner dimension.
        """
        self._update_rate_sensor = Sensor.float(
            "update-rate",
            description="The delay update rate",
            default=2.0,
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._update_rate_sensor)

        self._nbeams_sensor = Sensor.integer(
            "nbeams",
            description="Number of beams that this delay engine handles",
            default=0,
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._nbeams_sensor)

        self._antennas_sensor = Sensor.string(
            "antennas",
            description="JSON breakdown of the antennas (in KATPOINT format) associated with this delay engine",
            default="{}",
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._antennas_sensor)

        self._delays_sensor = Sensor.string(
            "delays",
            description="JSON object containing delays for each beam for each antenna at the current epoch",
            default="",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._delays_sensor)

    def start(self):
        super(DelayEngine, self).start()

    @request(Float())
    @return_reply()
    def request_set_update_rate(self, req, rate):
        """
        @brief    Set the update rate for delay calculations

        @param    rate  The update rate for recalculation of delay polynomials
        """
        self._update_rate_sensor.set_value(rate)
        # This should make a change to the beam manager object
        return ("ok",)


class FbfProductController(object):
    """
    Wrapper class for an FBFUSE product.
    """
    STATES = ["idle", "starting", "stopping", "capturing"]
    IDLE, STARTING, STOPPING, CAPTURING = STATES

    def __init__(self, parent, product_id, antennas, n_channels, streams, proxy_name, servers):
        """
        @brief      Construct new instance

        @param      parent            The parent FbfMasterController instance

        @param      product_id        The name of the product

        @param      antennas          A list of katpoint.Antenna objects

        @param      n_channels        The integer number of frequency channels provided by the CBF.

        @param      streams           A dictionary containing config keys and values describing the streams.

        @param      proxy_name        The name of the proxy associated with this subarray (used as a sensor prefix)

        @param      servers           A list of FbfWorkerServer instances allocated to this product controller
        """
        self._parent = parent
        self._product_id = product_id
        self._antennas = ",".join([_.name for _ in antennas])
        self._katpoint_antennas = antennas
        self._n_channels = n_channels
        self._streams = streams
        self._proxy_name = proxy_name
        self._servers = servers
        self._capturing = False
        self._beam_manager = None
        self._delay_engine = None
        self._managed_sensors = []
        self.setup_sensors()

    def __del__(self):
        self.teardown_sensors()

    def info(self):
        """
        @brief    Return a metadata dictionary describing this product controller
        """
        out = {
            "antennas":self._antennas,
            "nservers":len(self.servers),
            "capturing":self.capturing,
            "streams":self._streams,
            "nchannels":self._n_channels,
            "proxy_name":self._proxy_name
        }
        return out

    def add_sensor(self, sensor):
        """
        @brief    Add a sensor to the parent object

        @note     This method is used to wrap calls to the add_sensor method
                  on the parent FbfMasterController instance. In order to
                  disambiguate between sensors from describing different products
                  the associated proxy name is used as sensor prefix. For example
                  the "servers" sensor will be seen by clients connected to the
                  FbfMasterController server as "<proxy_name>-servers" (e.g.
                  "FBFUSE_1-servers").
        """
        prefix = "{}-".format(self._proxy_name)
        if sensor.name.startswith(prefix):
            self._parent.add_sensor(sensor)
        else:
            sensor.name = "{}{}".format(prefix,sensor.name)
            self._parent.add_sensor(sensor)
        self._managed_sensors.append(sensor)

    def setup_sensors(self):
        """
        @brief    Setup the default KATCP sensors.

        @note     As this call is made only upon an FBFUSE configure call a mass inform
                  is required to let connected clients know that the proxy interface has
                  changed.
        """
        self._state_sensor = Sensor.discrete(
            "state",
            description = "Denotes the state of this FBF instance",
            params = self.STATES,
            default = self.IDLE,
            initial_status = Sensor.UNKNOWN)
        self.add_sensor(self._state_sensor)

        self._available_antennas_sensor = Sensor.string(
            "available-antennas",
            description = "The antennas that are currently available for beamforming",
            default = self._antennas,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._available_antennas_sensor)

        self._cbc_nbeams_sensor = Sensor.integer(
            "coherent-beam-count",
            description = "The number of coherent beams that this FBF instance can currently produce",
            default = 400,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._cbc_nbeams_sensor)

        self._cbc_tscrunch_sensor = Sensor.integer(
            "coherent-beam-tscrunch",
            description = "The number time samples that will be integrated when producing coherent beams",
            default = 16,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._cbc_tscrunch_sensor)

        self._cbc_fscrunch_sensor = Sensor.integer(
            "coherent-beam-fscrunch",
            description = "The number frequency channels that will be integrated when producing coherent beams",
            default = 1,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._cbc_fscrunch_sensor)

        self._cbc_antennas_sensor = Sensor.string(
            " v",
            description = "The antennas that will be used when producing coherent beams",
            default = self._antennas,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._cbc_antennas_sensor)

        self._ibc_nbeams_sensor = Sensor.integer(
            "incoherent-beam-count",
            description = "The number of incoherent beams that this FBF instance can currently produce",
            default = 1,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._ibc_nbeams_sensor)

        self._ibc_tscrunch_sensor = Sensor.integer(
            "incoherent-beam-tscrunch",
            description = "The number time samples that will be integrated when producing incoherent beams",
            default = 16,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._ibc_tscrunch_sensor)

        self._ibc_fscrunch_sensor = Sensor.integer(
            "incoherent-beam-fscrunch",
            description = "The number frequency channels that will be integrated when producing incoherent beams",
            default = 1,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._ibc_fscrunch_sensor)

        self._ibc_antennas_sensor = Sensor.string(
            "incoherent-beam-antennas",
            description = "The antennas that will be used when producing incoherent beams",
            default = self._antennas,
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._ibc_antennas_sensor)

        self._servers_sensor = Sensor.string(
            "servers",
            description = "The server instances currently allocated to this product",
            default = ",".join(["{s.hostname}:{s.port}".format(s=server) for server in self._servers]),
            initial_status = Sensor.NOMINAL)
        self.add_sensor(self._servers_sensor)

        self._delay_engine_sensor = Sensor.string(
            "delay-engine",
            description = "The address of the delay engine serving this product",
            default = "",
            initial_status = Sensor.UNKNOWN)
        self.add_sensor(self._delay_engine_sensor)
        self._parent.mass_inform(Message.inform('interface-changed'))

    def teardown_sensors(self):
        """
        @brief    Remove all sensors created by this product from the parent server.

        @note     This method is required for cleanup to stop the FBF sensor pool
                  becoming swamped with unused sensors.
        """
        for sensor in self._managed_sensors:
            self._parent.remove_sensor(sensor)
        self._parent.mass_inform(Message.inform('interface-changed'))

    @property
    def servers(self):
        return self._servers

    @property
    def capturing(self):
        return self._state_sensor.value() == self.CAPTURING

    @property
    def idle(self):
        return self._state_sensor.value() == self.IDLE

    @property
    def starting(self):
        return self._state_sensor.value() == self.STARTING

    @property
    def stopping(self):
        return self._state_sensor.value() == self.STOPPING

    def _verify_antennas(self, antennas):
        """
        @brief      Verify that a set of antennas is available to this instance.

        @param      antennas   A CSV list of antenna names
        """
        antennas_set = set([ant.name for ant in self._katpoint_antennas])
        requested_antennas = set(antennas)
        return requested_antennas.issubset(antennas_set)

    def configure_coherent_beams(self, nbeams, antennas, fscrunch, tscrunch):
        """
        @brief      Set the configuration for coherent beams producted by this instance

        @param      nbeams          The number of beams that will be produced for the provided product_id

        @param      antennas        A comma separated list of physical antenna names. Only these antennas will be used
                                    when generating coherent beams (e.g. m007,m008,m009). The antennas provided here must
                                    be a subset of the antennas in the current subarray. If not an exception will be
                                    raised.

        @param      fscrunch        The number of frequency channels to integrate over when producing coherent beams.

        @param      tscrunch        The number of time samples to integrate over when producing coherent beams.
        """
        if not self.idle:
            raise Exception("Configuration calls can only be made on a product in an idle state")
        if not self._verify_antennas(antennas):
            raise Exception("Requested antennas are not a subset of the current subarray")
        self._cbc_nbeams_sensor.set_value(nbeams)
        #need a check here to determine if this is a subset of the subarray antennas
        self._cbc_fscrunch_sensor.set_value(fscrunch)
        self._cbc_tscrunch_sensor.set_value(tscrunch)
        self._cbc_antennas_sensor.set_value(antennas)

    def configure_incoherent_beam(self, antennas, fscrunch, tscrunch):
        """
        @brief      Set the configuration for incoherent beams producted by this instance

        @param      antennas        A comma separated list of physical antenna names. Only these antennas will be used
                                    when generating incoherent beams (e.g. m007,m008,m009). The antennas provided here must
                                    be a subset of the antennas in the current subarray. If not an exception will be
                                    raised.

        @param      fscrunch        The number of frequency channels to integrate over when producing incoherent beams.

        @param      tscrunch        The number of time samples to integrate over when producing incoherent beams.
        """
        if not self.idle:
            raise Exception("Configuration calls can only be made on a product in an idle state")
        if not self._verify_antennas(antennas):
            raise Exception("Requested antennas are not a subset of the current subarray")
        #need a check here to determine if this is a subset of the subarray antennas
        self._ibc_fscrunch_sensor.set_value(fscrunch)
        self._ibc_tscrunch_sensor.set_value(tscrunch)
        self._ibc_antennas_sensor.set_value(antennas)

    def _beam_to_sensor_string(self, beam):
        return beam.target.format_katcp()

    def start_beams(self):
        """
        @brief      Start the beamformer servers streaming

        @detail     This method evaluates the current configuration creates a new DelayEngine
                    and passes a start call to all allocated servers.
        """
        if not self.idle:
            raise Exception("Beams can only be started from an IDLE state")
        self._state_sensor.set_value(self.STARTING)
        self._beam_manager = BeamManager(self._cbc_nbeams_sensor.value(), self._cbc_antennas_sensor.value())
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

        self._delay_engine_sensor.set_value(self._delay_engine.bind_address)

        self._beam_sensors = []
        for beam in self._beam_manager.get_beams():
            sensor = Sensor.string(
                "coherent-beam-{}".format(beam.idx),
                description="R.A. (deg), declination (deg) and source name for coherent beam with ID {}".format(beam.idx),
                default=self._beam_to_sensor_string(beam),
                initial_status=Sensor.UNKNOWN)
            beam.register_observer(lambda beam, sensor=sensor:
                sensor.set_value(self._beam_to_sensor_string(beam)))
            self._beam_sensors.append(sensor)
            self.add_sensor(sensor)
        self._capturing = True
        self._state_sensor.set_value(self.CAPTURING)
        self._parent.mass_inform(Message.inform('interface-changed'))

    def stop_beams(self):
        """
        @brief      Stops the beamformer servers streaming.
        """
        if not self.capturing:
            return
        self._state_sensor.set_value(self.STOPPING)
        for server in self._servers:
            #yield server.req.deconfigure()
            pass
        self._state_sensor.set_value(self.IDLE)

    def add_beam(self, target):
        """
        @brief      Specify the parameters of one managed beam

        @param      target      A KATPOINT target object

        @return     Returns the allocated Beam object
        """
        if not self.capturing:
            raise Exception("Beam configurations should be specified after a call to start_beams")
        return self._beam_manager.add_beam(target)


    def add_tiling(self, target, number_of_beams, reference_frequency, overlap, epoch):
        """
        @brief   Add a tiling to be managed

        @param      target      A KATPOINT target object

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @returns    The created Tiling object
        """
        if not self.capturing:
            raise Exception("Tiling configurations should be specified after a call to start_beams")
        tiling = self._beam_manager.add_tiling(target, number_of_beams, reference_frequency, overlap)
        #tiling.generate(epoch)
        return tiling

    def add_dynamic_tiling(self, ra, dec, source_name, number_of_beams, reference_frequency, overlap, tolerance):
        """
        @brief   Create a new dynamic tiling object

        @param      target      A KATPOINT target object

        @param      reference_frequency     The reference frequency at which to calculate the synthesised beam shape,
                                            and thus the tiling pattern. Typically this would be chosen to be the
                                            centre frequency of the current observation.

        @param      overlap         The desired overlap point between beams in the pattern. The overlap defines
                                    at what power point neighbouring beams in the tiling pattern will meet. For
                                    example an overlap point of 0.1 corresponds to beams overlapping only at their
                                    10%-power points. Similarly a overlap of 0.5 corresponds to beams overlapping
                                    at their half-power points. [Note: This is currently a tricky parameter to use
                                    when values are close to zero. In future this may be define in sigma units or
                                    in multiples of the FWHM of the beam.]

        @param      tolerance       The tolerance criterion for triggering a re-tiling of the beam pattern. The tolerance
                                    is based around the percentage change in tiling efficiency. The tiling efficiency is
                                    calculated based on a combined measure of beam separation and covariance. If this all
                                    sounds nebulous and loose, thats because it is. A value of 0.1 corresponds to a 10%
                                    change in tiling efficency.

        @returns    The created DynamicTiling object
        """
        if not self.capturing:
            raise Exception("Tiling configurations should be specified after a call to start_beams")
        tiling = self._beam_manager.add_tiling(ra, dec, source_name, number_of_beams, reference_frequency, overlap, tolerance)
        #tiling.start_update_loop()
        return tiling

    def reset_beams(self):
        """
        @brief  reset and deallocate all beams and tilings managed by this instance

        @note   All tiling will be lost on this call and must be remade for subsequent observations
        """
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
    (opts, args) = parser.parse_args()
    FORMAT = "[ %(levelname)s - %(asctime)s - %(filename)s:%(lineno)s] %(message)s"
    logger = logging.getLogger('reynard')
    logging.basicConfig(format=FORMAT)
    logger.setLevel(opts.log_level.upper())
    logging.getLogger('katcp').setLevel('INFO')
    ioloop = tornado.ioloop.IOLoop.current()
    log.info("Starting FbfMasterController instance")
    server = FbfMasterController(opts.host, opts.port, dummy=opts.dummy)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))
    def start_and_display():
        server.start()
        log.info("Listening at {0}, Ctrl-C to terminate server".format(server.bind_address))

    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    main()


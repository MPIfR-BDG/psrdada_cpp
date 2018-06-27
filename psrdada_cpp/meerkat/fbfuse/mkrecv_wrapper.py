import threading
import os
import time
import sys
import logging
from collections import deque
from subprocess import Popen

log = logging.getLogger("psrdada_cpp.meerkat.fbfuse.MKRECVWrapper")
FORMAT = "[ %(levelname)s - %(asctime)s - %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)
log.setLevel(logging.INFO)

class RingBufferPipe(threading.Thread):
    def __init__(self, maxlen=10):
        threading.Thread.__init__(self)
        self.daemon = True
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.ring = deque(maxlen=maxlen)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, ''):
            self.ring.append(line)
        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)


class ProcessWrapper(object):
    def __init__(self):
        self._process = None
        self._stdout_pipe = RingBufferPipe()
        self._stderr_pipe = RingBufferPipe()

    def start(self, cmd):
        log.info("Starting process")
        self._process = Popen(cmd,
            stdout = self._stdout_pipe,
            stderr = self._stderr_pipe,
            shell = True)
        log.info("MKRECV process running with PID {}".format(self._process.pid))

    def status(self):
        print self._stdout_pipe.ring
        print self._stderr_pipe.ring

    def stop(self, timeout=10.0):
        log.info("Sending SIGTERM to MKRECV process")
        self._process.terminate()
        log.info("Waiting {} seconds for MKRECV to terminate...".format(timeout))
        now = time.time()
        while time.time()-now < timeout:
            retval = self._process.poll()
            if retval is not None:
                log.info("MKRECV returned a return value of {}".format(retval))
                return
            else:
                time.sleep(0.5)
        else:
            log.warning("MKRECV failed to terminate in allotted time")
            log.info("Killing MKRECV process")
            self._ingest_proc.kill()

if __name__ == "__main__":
    import signal
    x = MkrecvWrapper(None)
    x.start()
    def signal_handler(frame, signum):
        x.stop()
        sys.exit()
    signal.signal(signal.SIGINT, signal_handler)

    now = time.time()
    while time.time() - now < 5.0:
        x.status()
        time.sleep(1)
    x.stop()


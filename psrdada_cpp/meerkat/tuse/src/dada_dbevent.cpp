/***************************************************************************
 *  
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 * Attaches to in input data block as a viewer, and opens a socket to listen
 * for requests to write temporal events to the output data block. Can seek
 * back in time over cleared data blocks
 */

#include "ascii_header.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "node_array.h"
#include "multilog.h"
#include "diff_time.h"
#include "sock.h"
#include "tmutil.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <signal.h>

#include <sys/types.h>
#include <sys/socket.h>

#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <math.h>

using namespace std;

const size_t DADA_DBEVENT_DEFAULT_PORT= 30000;
const size_t DADA_DBEVENT_DEFAULT_INPUT_BUFFER = 80;
const size_t DADA_DBEVENT_DEFAULT_INPUT_DELAY= 60;
#define  DADA_DBEVENT_TIMESTR "%Y-%m-%d-%H:%M:%S"


int quit = 0;

typedef struct {

    // input HDU
    dada_hdu_t * in_hdu;  

    // output HDU
    dada_hdu_t * out_hdu;

    // multilog 
    multilog_t * log;

    // input data block's UTC_START
    time_t utc_start;

    // input data block's BYTES_PER_SECOND
    uint64_t bytes_per_second;

    time_t input_maximum_delay;

    char * header;

    size_t header_size;

    void * work_buffer;

    size_t work_buffer_size;

    uint64_t in_nbufs;

    uint64_t in_bufsz;

    uint64_t resolution;

    // verbosity
    int verbose;

} dada_dbevent_t;

typedef struct {

    uint64_t start_byte;

    uint64_t end_byte;

    float snr;

    float dm;

    float width;

    unsigned beam;

} event_t;

#define  DADA_DBEVENT_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

static int sort_events (const void *p1, const void *p2)
{
    event_t A = *(event_t *) p1;
    event_t B = *(event_t *) p2;
    if (A.start_byte < B.start_byte) return -1;
    if (A.start_byte > B.start_byte) return +1;
    return 0;
}

int check_read_offset (dada_dbevent_t * dbevent);
//int check_write_timestamps (dada_dbevent_t * dbevent);

int64_t calculate_byte_offset (dada_dbevent_t * dbevent, char * time_str_secs, char * time_str_frac);

int receive_events (dada_dbevent_t * dbevent, int listen_fd);

int dump_event(dada_dbevent_t * dbevent, double event_start_utc, double event_end_utc, float event_snr, float event_dm);

void usage();

void usage()
{
    fprintf (stdout,
            "dada_dbevent [options] inkey outkey\n"
            " inkey       input hexadecimal shared memory key\n"
            " outkey      input hexadecimal shared memory key\n"
            " -b percent  delay procesing of the input buffer up to this amount [default %d %%]\n"
            " -t delay    maximum delay (s) to retain data for [default %ds]\n"
            " -h          print this help text\n"
            " -p port     port to listen for event commands [default %d]\n"
            " -v          be verbose\n", 
            DADA_DBEVENT_DEFAULT_INPUT_BUFFER, 
            DADA_DBEVENT_DEFAULT_INPUT_DELAY, 
            DADA_DBEVENT_DEFAULT_PORT);
}

void signal_handler(int signalValue) 
{
    fprintf(stderr, "dada_dbevent: SIGINT/TERM\n");
    quit = 1;
}

double dm_delay(float DM, double freq1, double freq2)
{
    double freqghz1 = freq1/1e9;
    double freqghz2 = freq2/1e9;
    double delta_t  = 4.15 * 1e-3 * DM * ((1/pow(freqghz1,2)) - 1/(pow(freqghz2,2)));  // in seconds
    return delta_t;
}

int main (int argc, char **argv)
{
  // core dbevent data struct
  dada_dbevent_t dbevent = DADA_DBEVENT_INIT;

  // DADA Logger
  multilog_t* log = 0;

  // flag set in verbose mode
  char verbose = 0;

  // port to listen for event requests
  int port = DADA_DBEVENT_DEFAULT_PORT;

  // input hexadecimal shared memory key
  key_t in_dada_key;

  // output hexadecimal shared memory key
  key_t out_dada_key;

  float input_data_block_threshold = DADA_DBEVENT_DEFAULT_INPUT_BUFFER;

  int input_maximum_delay = DADA_DBEVENT_DEFAULT_INPUT_DELAY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hp:t:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        if (sscanf (optarg, "%f", &input_data_block_threshold) != 1)
        {
          fprintf (stderr, "dada_dbevent: could not parse input buffer level from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

      case 'h':
        usage();
        return EXIT_SUCCESS;

      case 'v':
        verbose++;
        break;

      case 'p':
        if (sscanf (optarg, "%d", &port) != 1) 
        {
          fprintf (stderr, "dada_dbevent: could not parse port from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

      case 't':
        if (sscanf (optarg, "%d", &input_maximum_delay) != 1)
        {
          fprintf (stderr, "dada_dbevent: could not parse maximum input delay from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

      default:
        usage ();
        return EXIT_SUCCESS;
    }
  }

  if (argc - optind != 2)
  { 
    fprintf (stderr, "dada_dbevent: expected 2 command line arguments\n");
    usage();
    return EXIT_FAILURE;
  }

  if (sscanf (argv[optind], "%x", &in_dada_key) != 1) 
  {
    fprintf (stderr,"dada_dbevent: could not parse in_key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (sscanf (argv[optind+1], "%x", &out_dada_key) != 1) 
  {
    fprintf (stderr,"dada_dbevent: could not parse out_key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  // install some signal handlers
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  log = multilog_open ("dada_dbevent", 0);
  multilog_add (log, stderr);

  dbevent.verbose = verbose;
  dbevent.log = log;
  dbevent.input_maximum_delay = (time_t) input_maximum_delay;
  dbevent.work_buffer_size = 1024 * 1024; 
  dbevent.work_buffer =  malloc (dbevent.work_buffer_size);
  if (!dbevent.work_buffer)
  {
    multilog(log, LOG_INFO, "could not allocate memory for work buffer\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog(log, LOG_INFO, "connecting to data blocks\n");

  dbevent.in_hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbevent.in_hdu, in_dada_key);
  if (dada_hdu_connect (dbevent.in_hdu) < 0)
  {
    multilog(log, LOG_ERR, "could not connect to input HDU\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read(dbevent.in_hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not open input HDU as viewer\n");
    return EXIT_FAILURE;
  }

  dbevent.out_hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbevent.out_hdu, out_dada_key);
  if (dada_hdu_connect (dbevent.out_hdu) < 0)
  {
    multilog(log, LOG_ERR, "could not connect to output HDU\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write (dbevent.out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not open output HDU as writer\n");
    return EXIT_FAILURE;
  }

  // open listening socket
  if (verbose)
    multilog (log, LOG_INFO, "main: sock_create(%d)\n", port);
  int listen_fd = sock_create (&port);
  if (listen_fd < 0)
  { 
    multilog (log, LOG_ERR, "could not open socket: %s\n", strerror(errno));
    quit = 2;
  }
  else
  {
    if (verbose)
      multilog (log, LOG_INFO, "listening on port %d for dump requests\n", port);
  }


  fd_set fds;
  struct timeval timeout;
  int fds_read;
  
  // now get the header from the input data block
  if (verbose)
    multilog(log, LOG_INFO, "waiting for input header\n");
  if (dada_hdu_open (dbevent.in_hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not get input header\n");
    quit = 1;
  }
  else
  {
    if (verbose > 1)
    {
      fprintf (stderr, "==========\n");
      fprintf (stderr, "%s", dbevent.in_hdu->header);
      fprintf (stderr, "==========\n");
    }

    dbevent.header_size = ipcbuf_get_bufsz (dbevent.in_hdu->header_block);
    dbevent.header = (char *) malloc (dbevent.header_size);
    if (!dbevent.header)
    {
      multilog (log, LOG_ERR, "failed to allocate memory for header\n");
      quit = 1;
    }
    else
    {
      // make a local copy of the header
      memcpy (dbevent.header, dbevent.in_hdu->header, dbevent.header_size);

      // immediately mark this header as cleared
      ipcbuf_mark_cleared (dbevent.in_hdu->header_block);

      char utc_buffer[64];
      // get the UTC_START and TSAMP / BYTES_PER_SECOND for this observation    
      if (ascii_header_get (dbevent.header, "UTC_START", "%s", utc_buffer) < 0) 
      {
        multilog (log, LOG_ERR, "could not extract UTC_START from input datablock header\n");
        quit = 2;
      }
      else
      {
        if (verbose)
          multilog(log, LOG_INFO, "input UTC_START=%s\n", utc_buffer);
        dbevent.utc_start = str2utctime (utc_buffer);
        if (dbevent.utc_start == (time_t)-1) 
        {
          multilog (log, LOG_ERR, "could not parse UTC_START from '%s'\n", utc_buffer);
          quit = 2;
        }
      }

      if (ascii_header_get (dbevent.header, "BYTES_PER_SECOND", "%" PRIu64, &(dbevent.bytes_per_second)) < 0) 
      {
        multilog (log, LOG_ERR, "could not extract BYTES_PER_SECOND from input datablock header\n");
        quit = 2;
      }
      else
      {
        if (verbose)
          multilog(log, LOG_INFO, "input BYTES_PER_SECOND=%" PRIu64"\n", dbevent.bytes_per_second);
      }

      if (ascii_header_get (dbevent.header, "RESOLUTION", "%" PRIu64, &(dbevent.resolution)) < 0)
        dbevent.resolution = 1;

      if (verbose)
        multilog(log, LOG_INFO, "input RESOLUTION=%" PRIu64"\n", dbevent.resolution);
    }
  }

  ipcbuf_t * db = (ipcbuf_t *) dbevent.in_hdu->data_block;

  // get the number and size of buffers in the input data block
  dbevent.in_nbufs = ipcbuf_get_nbufs (db);
  dbevent.in_bufsz = ipcbuf_get_bufsz (db);

  while (!quit)
  {
    // setup file descriptor set for listening
    FD_ZERO(&fds);
    FD_SET(listen_fd, &fds);
    timeout.tv_sec = 0;
    timeout.tv_usec = 1000000;
    fds_read = select(listen_fd+1, &fds, (fd_set *) 0, (fd_set *) 0, &timeout);

    // problem with select call
    if (fds_read < 0)
    {
      multilog (log, LOG_ERR, "select failed: %s\n", strerror(errno));
      quit = 2;
      break;
    }
    // select timed out, check input HDU for end of data
    else if (fds_read == 0)
    {
      if (verbose > 1)
        multilog (log, LOG_INFO, "main: check_read_offset ()\n");
      int64_t n_skipped = check_read_offset (&dbevent);
      if (n_skipped < 0)
        multilog (log, LOG_WARNING, "check_db_times failed\n");
      if (verbose > 1)
        multilog (log, LOG_INFO, "main: check_db_times skipped %" PRIi64" events\n", n_skipped);
    }
    // we received a new connection on our listening FD, process comand
    else
    {
      if (verbose)
        multilog (log, LOG_INFO, "main: receiving events on socket\n");
      int events_recorded = receive_events (&dbevent, listen_fd);
      if (events_recorded < 0)
      {
        multilog (log, LOG_INFO, "main: quit requested via socket\n");
        quit = 1;
      }
      if (verbose)
        multilog (log, LOG_INFO, "main: received %d events\n", events_recorded);
    }

    // check how full the input datablock is
    float percent_full = ipcio_percent_full (dbevent.in_hdu->data_block) * 100;
    if (verbose > 1)
      multilog (log, LOG_INFO, "input datablock %5.2f percent full\n", percent_full);

    int64_t read_offset, remainder, seek_byte, seeked_byte;

    while (!quit && percent_full > input_data_block_threshold)
    {
      if (verbose)
        multilog (log, LOG_INFO, "percent_full=%5.2f threshold=%5.2f\n", percent_full, input_data_block_threshold);
      // since we are too full, seek forward 1 block 
      read_offset = ipcio_tell (dbevent.in_hdu->data_block);

      // if the current read offset is a full block, make it so
      remainder = read_offset % dbevent.in_bufsz;

      if (remainder != 0)
        seek_byte = (int64_t) dbevent.in_bufsz - remainder;
      else
        seek_byte = (int64_t) dbevent.in_bufsz;

      if ((seek_byte < 0) || (seek_byte > dbevent.in_bufsz))
        multilog (log, LOG_WARNING, "main: seek_byte limits warning: %" PRIi64"\n", seek_byte);

      if (verbose)
        multilog (log, LOG_INFO, "main: ipcio_seek(%" PRIu64", SEEK_CUR)\n", seek_byte);

      // seek forward (from curr pos) to the next full block boundary
      seeked_byte = ipcio_seek (dbevent.in_hdu->data_block, seek_byte, SEEK_CUR);
      if (seeked_byte < 0)
      {
        multilog (log, LOG_INFO, "main: ipcio_seek failed\n");
        quit = 1; 
      }
 
      // sleep a short space to allow writer to write!
      usleep(10000);

      // update the percentage full for the data block
      percent_full = ipcio_percent_full (dbevent.in_hdu->data_block) * 100;
      if (verbose)
        multilog (log, LOG_INFO, "input datablock reduced to %5.2f percent full\n", percent_full);
    }

    if (ipcbuf_eod (db))
    {
      if (verbose)
        multilog (log, LOG_INFO, "EOD now true\n");
      quit = 1;
    }
  }

  if (quit)
  {
    if (!ipcbuf_eod (db))
    {
      multilog (log, LOG_INFO, "quit requested, trying to clear all remaining data blocks\n");
      uint64_t bytes_written = ipcbuf_get_write_byte_xfer ((ipcbuf_t *) dbevent.in_hdu->data_block);
      multilog (log, LOG_INFO, "quit requested: total bytes written=%" PRIu64"\n", bytes_written);
      int64_t bytes_seeked = ipcio_seek (dbevent.in_hdu->data_block, (int64_t) bytes_written, SEEK_SET);
      multilog (log, LOG_INFO, "quit requested: seeked to byte %" PRIi64"\n", bytes_seeked);
    }
  }

  free (dbevent.work_buffer);
  if (dbevent.header)
    free (dbevent.header);
  dbevent.header = 0;

  if (dada_hdu_disconnect (dbevent.in_hdu) < 0)
  {
    fprintf (stderr, "dada_dbevent: disconnect from input data block failed\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_unlock_write (dbevent.out_hdu) < 0)
  {
    fprintf (stderr, "dada_dbevent: unlock write on output data block failed\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_disconnect (dbevent.out_hdu) < 0)
  {
    fprintf (stderr, "dada_dbevent: disconnect from output data block failed\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


int check_read_offset (dada_dbevent_t * dbevent)
{
  const uint64_t max_delay_bytes = dbevent->input_maximum_delay * dbevent->bytes_per_second;

  unsigned have_old_buffers = 1;

  // check for end of data before doing anything
  if (ipcbuf_eod ((ipcbuf_t *) dbevent->in_hdu->data_block))
    have_old_buffers = 0;

  // get the time and byte offset for the time
  const time_t now = time(0);
  const time_t obs_offset = (now - dbevent->utc_start);
  if (obs_offset <= 0)
  {
    if (dbevent->verbose)
      multilog (dbevent->log, LOG_INFO, "check_read_offset: now <= obs_offset\n");
    return (0);
  }

  if (dbevent->verbose)
    multilog (dbevent->log, LOG_INFO, "check_read_offset: now=%ld utc_start=%ld obs_offset=%ld\n", now, dbevent->utc_start, obs_offset);
  const uint64_t now_byte_offset = (uint64_t) obs_offset * dbevent->bytes_per_second;

  uint64_t read_offset, remainder, seek_byte;
  int64_t seeked_byte;
  unsigned n_skipped = 0;

  while (have_old_buffers)
  {
    // get the current read offset in bytes
    read_offset = ipcio_tell (dbevent->in_hdu->data_block);

    // if the current read offset + the maximum allowed delay is still less than 
    // the current byte offset, then we MUST read while it is not
    if (read_offset + max_delay_bytes < now_byte_offset)
    {
      if (dbevent->verbose)
        multilog (dbevent->log, LOG_INFO, "check_read_offset: read_offset[%" PRIu64"] + max_delay_bytes[%" PRIu64"] < now_byte_offset[%" PRIu64"]\n", read_offset, max_delay_bytes, now_byte_offset);

      remainder = read_offset % dbevent->in_bufsz;
      seek_byte = 0;

      if (dbevent->verbose > 1)
        multilog (dbevent->log, LOG_INFO, "check_read_offset: read_offset=%" PRIu64" remainder=%" PRIu64"\n", read_offset, remainder);

      // check for a partially read block
      if (remainder != 0)
      {
        seek_byte = dbevent->in_bufsz - remainder;
      }
      // otherwise we will seek forward 1 full block
      else
      {
        seek_byte = dbevent->in_bufsz;
      }

      if ((seek_byte < 0) || (seek_byte > dbevent->in_bufsz))
      {
        multilog (dbevent->log, LOG_WARNING, "check_read_offset: seek_byte limits warning: %" PRIi64"\n", seek_byte);
      }

      if (dbevent->verbose)
        multilog (dbevent->log, LOG_INFO, "check_read_offset: ipcio_seek(%" PRIu64", SEEK_CUR)\n", seek_byte);
      seeked_byte = ipcio_seek (dbevent->in_hdu->data_block, seek_byte, SEEK_CUR);
      if (seeked_byte < 0)
      {
        multilog (dbevent->log, LOG_INFO, "check_read_offset: ipcio_seek failed\n");
        return -1;
      }
      n_skipped ++;
    }
    else
      have_old_buffers = 0;
  
    // also check for end of data
    if (ipcbuf_eod ((ipcbuf_t *) dbevent->in_hdu->data_block))
    {
      have_old_buffers = 0;
    }
  }

  if (dbevent->verbose)
    multilog (dbevent->log, LOG_INFO, "check_read_offset: skipped %" PRIi64" blocks\n", n_skipped);

  return n_skipped;
}

int receive_events (dada_dbevent_t * dbevent, int listen_fd)
{
  multilog_t * log = dbevent->log;

  int    fd     = 0;
  FILE * sockin = 0;
  unsigned more_events = 1;

  unsigned buffer_size = 1024;
  char buffer[buffer_size];

  char * event_start;
  char * event_start_fractional;
  char * event_end;
  char * event_end_fractional;
  char * event_snr_str;
  char * event_dm_str;
  char * event_width_str;
  char * event_beam_str;

  // arrays for n_events
  uint64_t  n_events = 0;
  event_t * events = NULL;

  int events_recorded = 0;
  int events_missed = 0;

  if (dbevent->verbose)
    multilog (log, LOG_INFO, "receive_events: sock_accept(listen_fd)\n");
  fd = sock_accept (listen_fd);
  if (fd < 0)
  {
    multilog(log, LOG_WARNING, "error accepting connection %s\n", strerror(errno));
    return -1;
  }

  sockin = fdopen(fd,"r");
  if (!sockin)
  {
    multilog(log, LOG_WARNING, "error creating input stream %s\n", strerror(errno));
    close(fd);
    return -1;
  }

  setbuf (sockin, 0);

  // first line on the socket should be the number of events
  fgets (buffer, buffer_size, sockin);
  if (sscanf (buffer, "N_EVENTS %" PRIu64, &n_events) != 1)
  {
    multilog(log, LOG_WARNING, "failed to parse N_EVENTS\n");
    more_events = 0;  
  }
  else
  {
    events = (event_t *) malloc (sizeof(event_t) * n_events);
  }

  // second line on the socket should be the UTC_START of the obsevation
  fgets (buffer, buffer_size, sockin);
  time_t event_utc_start = str2utctime (buffer);

  char * comment = 0;
  unsigned i = 0;
  const char * sep_time = ". \t";
  const char * sep_float = " \t";
  int64_t offset;
  uint64_t remainder;
  
  while (more_events > 0 && !feof(sockin))
  {
    if (dbevent->verbose > 1) 
      multilog (log, LOG_INFO, "getting new line\n");
    char * saveptr = 0;

    fgets (buffer, buffer_size, sockin);

    //if (dbevent->verbose > 1)
    multilog (log, LOG_INFO, " <- %s", buffer);

    // ignore comments
    comment = strchr( buffer, '#' );
    if (comment)
      *comment = '\0';

    comment = strchr( buffer, '\r' );
    if (comment)
      *comment = '\0';

    if (dbevent->verbose)
       multilog (log, LOG_INFO, "< - %s\n", buffer);

    if (strlen(buffer) < 10)
      continue;

    if (strcmp(buffer, "QUIT") == 0)
    {
      multilog (log, LOG_WARNING, "receive_events: QUIT event received\n");
      more_events = -1;
      continue;
    }

    // extract START_UTC string excluding sub-second components
    event_start = strtok_r (buffer, sep_time, &saveptr);
    if (event_start == NULL)
    {
      multilog (log, LOG_WARNING, "receive_events: problem extracting event_start\n");
      more_events = 0;
      continue;
    }
    event_start_fractional = strtok_r (NULL, sep_time, &saveptr);

    if (dbevent->verbose)
      multilog (log, LOG_INFO, "event_start=%s event_start_fractional=%s\n", event_start, event_start_fractional);

    offset = calculate_byte_offset (dbevent, event_start, event_start_fractional);
    if (offset >= 0)
    {
      remainder = offset % dbevent->resolution;
      if (remainder != 0)
      {
        if (offset > dbevent->resolution)
          events[i].start_byte = (uint64_t) (offset - remainder);
        else
          events[i].start_byte = dbevent->resolution;
      }
      else
        events[i].start_byte = (uint64_t) offset;
    }
    else
      events[i].start_byte = 0;

    // extract END_UTC string excluding sub-second components
    event_end = strtok_r (NULL, sep_time, &saveptr);
    if (event_end == NULL)
    {
      multilog (log, LOG_WARNING, "receive_events: problem extracting event_end\n");
      more_events = 0;
      continue;
    }
    event_end_fractional = strtok_r (NULL, sep_time, &saveptr);

    if (dbevent->verbose)
      multilog (log, LOG_INFO, "event_end=%s event_end_fractional=%s\n", event_start, event_start_fractional);
    offset = calculate_byte_offset (dbevent, event_end, event_end_fractional);
    if (offset >= 0)
    {
      events[i].end_byte = (uint64_t) offset;
      remainder = offset % dbevent->resolution;
      if (remainder != 0)
        events[i].end_byte += (dbevent->resolution - remainder);
    }
    else
      events[i].end_byte = 0;

    event_dm_str = strtok_r (NULL, sep_float, &saveptr);
    sscanf(event_dm_str, "%f", &(events[i].dm));

    event_snr_str = strtok_r (NULL, sep_float, &saveptr);
    sscanf(event_snr_str, "%f", &(events[i].snr));

    event_width_str = strtok_r (NULL, sep_float, &saveptr);
    sscanf(event_width_str, "%f", &(events[i].width));

    event_beam_str = strtok_r (NULL, sep_float, &saveptr);
    sscanf(event_beam_str, "%u", &(events[i].beam));

    if (dbevent->verbose)
      multilog (dbevent->log, LOG_INFO, "event: %" PRIi64" - %" PRIi64" SNR=%f, DM=%f WIDTH=%f beam=%u\n",
                events[i].start_byte, events[i].end_byte, events[i].snr, events[i].dm,
                events[i].width, events[i].beam);

    i++;

    if (i >= n_events)
      more_events = 0;
  }

  if (n_events > 0)
  {
    if (event_utc_start != dbevent->utc_start)
    {
      multilog (dbevent->log, LOG_WARNING, "Event UTC_START [%d] != Obs UTC_START [%d]\n", event_utc_start, dbevent->utc_start);
    }
    else
    {
      // sort the events based on event start time
      qsort (events, n_events, sizeof (event_t), sort_events);

      // now check for overlapping events
      for (i=1; i<n_events; i++)
      {
        // start overlap
        if (events[i].start_byte < events[i-1].end_byte)
        {
          if (dbevent->verbose)
            multilog (dbevent->log, LOG_INFO, "amalgamating event idx %d into %d\n", i-1, i);
          events[i].start_byte = events[i-1].start_byte;
          events[i-1].start_byte = 0;

          if (events[i-1].end_byte > events[i].end_byte)
            events[i].end_byte = events[i-1].end_byte;
          events[i-1].end_byte = 0;
        }
      }

      // for each event, check that its in the future, and if so, seek forward to it
      uint64_t current_byte = 0;
      int64_t seeked_byte = 0;
      for (i=0; i<n_events; i++)
      {
        if ((events[i].start_byte == 0) && (events[i].end_byte == 0))
        {
          if (dbevent->verbose)
            multilog (dbevent->log, LOG_INFO, "ignoring event[%d], start_byte == end_byte == 0\n", i);
          continue;
        }

        current_byte = ipcio_tell (dbevent->in_hdu->data_block);
        //multilog (dbevent->log, LOG_INFO, "current_byte=%"PRIu64"\n", current_byte);

        if (events[i].start_byte < current_byte)
        {
          multilog (dbevent->log, LOG_WARNING, "skipping events[%d], current_byte [%" PRIu64"] past event start_byte [%" PRIu64"]\n", i, current_byte, events[i].start_byte);
          events_missed++;
          continue;
        }

        // seek forward to the relevant point in the datablock
        if (dbevent->verbose)
          multilog (dbevent->log, LOG_INFO, "seeking forward %" PRIu64" bytes from start of obs\n", events[i].start_byte);
        seeked_byte = ipcio_seek (dbevent->in_hdu->data_block, (int64_t) events[i].start_byte, SEEK_SET);
        if (seeked_byte < 0)
        {
          multilog (dbevent->log, LOG_WARNING, "could not seek to byte %" PRIu64"\n", events[i].start_byte);
          events_missed++;
          continue;
        }

        if (dbevent->verbose)
          multilog (dbevent->log, LOG_INFO, "seeked_byte=%" PRIi64"\n", seeked_byte);

        // determine how much to read
        size_t to_read = events[i].end_byte - events[i].start_byte;
        multilog (dbevent->log, LOG_INFO, "to read = %d [%" PRIu64" - %" PRIu64"]\n", to_read, events[i].end_byte, events[i].start_byte);

        if (dbevent->work_buffer_size < to_read)
        {
          dbevent->work_buffer_size = to_read;
          if (dbevent->verbose)
            multilog (dbevent->log, LOG_INFO, "reallocating work_buffer [%p] to %d bytes\n", 
                      dbevent->work_buffer, dbevent->work_buffer_size);
          dbevent->work_buffer = realloc (dbevent->work_buffer, dbevent->work_buffer_size);
          if (dbevent->verbose)
            multilog (dbevent->log, LOG_INFO, "reallocated work_buffer [%p]\n", dbevent->work_buffer);
        }
         
        // read the event from the input buffer 
        if (dbevent->verbose)
          multilog (dbevent->log, LOG_INFO, "reading %d bytes from input HDU into work buffer\n", to_read);
        ssize_t bytes_read = ipcio_read (dbevent->in_hdu->data_block, (char *) dbevent->work_buffer, to_read);
        if (dbevent->verbose)
          multilog (dbevent->log, LOG_INFO, "read %d bytes from input HDU into work buffer\n", bytes_read);
        if (bytes_read < 0)
        {
          multilog (dbevent->log, LOG_WARNING, "receive_events: ipcio_read on input HDU failed\n");
          return -1;
        }

        events_recorded++;

        char * header = ipcbuf_get_next_write (dbevent->out_hdu->header_block);
        uint64_t header_size = ipcbuf_get_bufsz (dbevent->out_hdu->header_block);
        if (header_size < dbevent->header_size)
        {
          multilog (log, LOG_ERR, "receive_events: output header too small for input header\n");
          return -1;
        }

        // copy the input header to the output
        memcpy (header, dbevent->header, dbevent->header_size);

        // now write some relevant data to the header
        ascii_header_set (header, "OBS_OFFSET", "%" PRIu64, events[i].start_byte);
        ascii_header_set (header, "FILE_SIZE", "%ld", to_read);
        ascii_header_set (header, "EVENT_SNR", "%f", events[i].snr);
        ascii_header_set (header, "EVENT_DM", "%f",  events[i].dm);
        ascii_header_set (header, "EVENT_WIDTH", "%f",  events[i].width);
        ascii_header_set (header, "EVENT_BEAM", "%u",  events[i].beam);
        //ascii_header_set (header, "EVENT_FIRST_FREQUENCY", "%f", events[i].f1);
        //ascii_header_set (header, "EVENT_LAST_FREQUENCY", "%f", events[i].f2)

        // tag this header as filled
        ipcbuf_mark_filled (dbevent->out_hdu->header_block, header_size);

        //TBD: Convert the written amount to the actual amount of data required
        // Will need DM, nchans, Centre Frequency
        // How much time before the event? How much after? the data after at least > time before for the snapped candidate.
        
        /*char* event_buffer = NULL;
        // offset calcluation here
        event_buffer = malloc(event_bytes * sizeof(char));
        int numbytes  = ((dm_delay(events[i].dm, events[i].f1, events[i].f2) /tsamp) + x) * nchans * 2  // 16 bytes per sample
        int ii ;
        for (ii = 0 ; ii < nchans; ++ii)
        {
            // Find where the start byte is , seek the pointer to it
            // memcpy to event_buffer with numbytes
            // seek again to end of the time samples for that channel
        }*/

        // write the specified amount to the output data block
        ipcio_write (dbevent->out_hdu->data_block, (char *) dbevent->work_buffer, to_read);

        // close the data block to ensure EOD is written
        if (dada_hdu_unlock_write (dbevent->out_hdu) < 0)
        {
          multilog (log, LOG_ERR, "could not close output HDU as writer\n");
          return -1; 
        }

        // lock write again to re-open for the next event
        if (dada_hdu_lock_write (dbevent->out_hdu) < 0)
        {
          multilog (log, LOG_ERR, "could not open output HDU as writer\n");
          return -1;
        }
      }

      multilog (dbevent->log, LOG_INFO, "recorded=%d missed=%d\n", events_recorded, events_missed);
    }
  }

  fclose(sockin);
  close (fd);

  if (events)
    free(events);
  events = 0;

  return events_recorded;
}

int64_t calculate_byte_offset (dada_dbevent_t * dbevent, char * time_str_secs, char * time_str_frac)
{
  time_t   time_secs;         // integer time in seconds
  uint64_t time_frac_numer;   // numerator of fractional time
  uint64_t time_frac_denom;   // denominator of fractional time

  int64_t  event_byte_offset = -1;
  uint64_t event_byte;
  uint64_t event_byte_frac;
 
  time_secs = str2utctime (time_str_secs);
  sscanf (time_str_frac, "%" PRIu64, &time_frac_numer);
  time_frac_denom = (uint64_t) powf(10,strlen(time_str_frac));

  if (dbevent->verbose > 1)
    multilog (dbevent->log, LOG_INFO, "calculate_byte_offset: time_secs=%d, time_frac_numer=%" PRIu64", time_frac_denom=%" PRIu64"\n", 
              time_secs, time_frac_numer, time_frac_denom);

  // check we have utc_start and that this event is in the future
  if (dbevent->utc_start && (time_secs >= dbevent->utc_start))
  {
    event_byte = (time_secs - dbevent->utc_start) * dbevent->bytes_per_second;
    event_byte_frac = time_frac_numer * dbevent->bytes_per_second;
    event_byte_frac /= time_frac_denom;
    event_byte_offset = (int64_t) event_byte + event_byte_frac;
    if (dbevent->verbose > 1)
      multilog (dbevent->log, LOG_INFO, "calculate_byte_offset: byte_offset [%" PRIi64"]=event_byte [%" PRIu64"] + event_byte_frac [%" PRIu64"]\n",
                event_byte_offset, event_byte, event_byte_frac);
  }
  else
  {
    multilog (dbevent->log, LOG_ERR, "calculate_byte_offset: time_secs=%d >= dbevent->utc_start=%" PRIu64"\n", time_secs, dbevent->utc_start);
  }
  return event_byte_offset;
}


/*
 * dump the specified event to the output datablock */
int dump_event (dada_dbevent_t * dbevent, double event_start_utc, 
                double event_end_utc, float event_snr, float event_dm)
{
  multilog (dbevent->log, LOG_INFO, "event time: %lf - %lf [seconds]\n", event_start_utc, event_end_utc);
  multilog (dbevent->log, LOG_INFO, "event SNR: %f\n", event_snr);
  multilog (dbevent->log, LOG_INFO, "event DM: %f\n", event_dm);
  return 0;
}

#!/usr/bin/env python
#
# Plot data of gated spectrometer file output
#
import argparse
import pylab as plt
import numpy
import logging
import coloredlogs



parser = argparse.ArgumentParser(description="Plot gated spectrometer output")
parser.add_argument('filename', nargs=1)
parser.add_argument('-v', action='store_true')
parser.add_argument('--log-level',dest='log_level',type=str,
        help='Logging level',default="INFO")

args = parser.parse_args()

log = logging.getLogger("GatedPlotter")
coloredlogs.install(
    fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
    level=args.log_level.upper(),
    logger=log)



def getSingleSpectrum(rawData, bitDepth):
    if bitDepth == 8:
        data = numpy.fromstring(rawData, dtype='int8')
    elif bitDepth == 32:
        data = numpy.fromstring(rawData, dtype='float32')
    else:
        print "WTF"
    return numpy.asarray(data, dtype='float')



with open(args.filename[0], 'rb') as inputfile:

    HEADER = inputfile.read(4096)
    print(HEADER)

    keys = ["fft_length", "naccumulate", "output_bit_depth", "full_stokes_output"]
    config = {}

    for line in HEADER.split('\n'):
        line = line.split('#')[0]
        for key in keys:
            if key in line:
                config[key] = line.split()[1].strip().rstrip('\x00')
                log.debug("Found item in header: {} -> {}".format([key], config[key]))

    fft_length = int(config["fft_length"])
    naccumulate = int(config["naccumulate"])
    bitDepth = int(config["output_bit_depth"])

    nChannels = fft_length/2 + 1

    rawData = inputfile.read()
    if config["full_stokes_output"] ==  "no":
        size_of_line = 2 * (nChannels * bitDepth / 8 + 64 / 8)
    else:
        size_of_line = 8 * (nChannels * bitDepth / 8 + 64 / 8)

    n_rows = len(rawData) / size_of_line
    log.info("Found {} on/off spectra in file".format(n_rows) )

    rem = len(rawData) % size_of_line
    if rem > 0:
        log.warning("{} bytes remaining in datafile!".format(rem))

    if config["full_stokes_output"] ==  "no":

        D0 = numpy.zeros([n_rows, nChannels])
        D0N = numpy.zeros(n_rows)
        D1 = numpy.zeros([n_rows, nChannels])
        D1N = numpy.zeros(n_rows)

        for i in range(n_rows):
            start = i * size_of_line
            D0[i] = getSingleSpectrum(rawData[start: start + nChannels * bitDepth / 8], bitDepth)
            D0N[i] = numpy.fromstring(rawData[start + nChannels * bitDepth / 8:][:8], dtype='uint64')

            start += size_of_line/2
            D1[i] = getSingleSpectrum(rawData[start: start + nChannels * bitDepth / 8], bitDepth)
            D1N[i] = numpy.fromstring(rawData[start + nChannels * bitDepth / 8:][:8], dtype='uint64')

        fig = plt.figure()
        s1 = fig.add_subplot(121)
        s1.imshow(10*numpy.log10(D0 + 1E-32))
        s2 = fig.add_subplot(122)
        s2.imshow(10*numpy.log10(D1+1E-32))

        fig = plt.figure()
        s1 = fig.add_subplot(121)
        s1.set_title('Noise Diode Off')
        s1.plot(10*numpy.log10(D0.sum(axis=0)[1:] + 1E-32))
        s1.set_ylabel('10 * log10(Power) [a.u.]')
        s1.set_xlabel('Channel')

        s2 = fig.add_subplot(122)
        s2.set_title('Noise Diode On')
        s2.plot(10*numpy.log10(D1.sum(axis=0)[1:] + 1E-32))
        s1.set_ylabel('10 * log10(Power) [a.u.]')
        s2.set_xlabel('Channel')

        fig.suptitle(args.filename)

        plt.show()

    elif config["full_stokes_output"] ==  "yes":

        D_on = numpy.zeros([4, n_rows, nChannels])
        D_off = numpy.zeros([4, n_rows, nChannels])

        N_on = numpy.zeros(n_rows)
        N_off = numpy.zeros(n_rows)

        size_of_spectrum = nChannels * bitDepth / 8
        start_of_spectrum = 0
        end_of_spectrum = start_of_spectrum + size_of_spectrum

        for i in range(n_rows):
            log.debug("Line {}:".format(i))
            for j in range(4):

                log.debug("  - j = {} off: [{}:{}]".format(j, start_of_spectrum, end_of_spectrum))
                D_off[j,i] = getSingleSpectrum(rawData[start_of_spectrum:end_of_spectrum], bitDepth)
                N_off[i] = numpy.fromstring(rawData[end_of_spectrum:end_of_spectrum+8], dtype='uint64')
                log.debug("    P = {}, N = {}".format(sum(D_off[j,i]), N_off[i]))

                start_of_spectrum = end_of_spectrum + 64/8
                end_of_spectrum = start_of_spectrum + size_of_spectrum

                log.debug("  - j = {}, on: [{}:{}]".format(j, start_of_spectrum, end_of_spectrum))
                D_on[j,i] = getSingleSpectrum(rawData[start_of_spectrum:end_of_spectrum], bitDepth)
                N_on[i] = numpy.fromstring(rawData[end_of_spectrum:end_of_spectrum+8], dtype='uint64')
                log.debug("    P = {}, N = {}".format(sum(D_on[j,i]), N_on[i]))

                start_of_spectrum = end_of_spectrum + 64/8
                end_of_spectrum = start_of_spectrum + size_of_spectrum

        fig = plt.figure()
        s1 = fig.add_subplot(241)
        s1.imshow(10*numpy.log10(D_off[0] + 1E-32))
        s2 = fig.add_subplot(245)
        s2.imshow(10*numpy.log10(D_on[0] + 1E-32))

        for j, t in  enumerate(['Q','U', 'V']):
            s1 = fig.add_subplot(241 + 1 + j)
            #s1.imshow(D_off[j+1])
            s1.imshow(D_off[j+1] / (D_off[0] + 1E-32))
            s2 = fig.add_subplot(245 + 1 + j)
            #s2.imshow(D_on[j+1])
            s2.imshow(D_on[j+1] /(D_on[0] + 1E-32))

        fig = plt.figure()
        s1 = fig.add_subplot(221)

        Ioff = D_off[0].sum(axis=0)
        Ion = D_on[0].sum(axis=0)

        s1.plot(10*numpy.log10(Ioff[1:] + 1E-32), label='Off')
        s1.plot(10*numpy.log10(Ion[1:] + 1E-32), label='On')
        s1.set_ylabel('10 * log10(Power) [a.u.]')
        s1.set_xlabel('Channel')

        for j, t in  enumerate(['Q','U', 'V']):
            s = fig.add_subplot(2,2,j+2)

            Xoff = D_off[j+1].sum(axis=0) / Ioff
            Xon = D_on[j+1].sum(axis=0) / Ion
            s.plot(Xoff, label='Off')
            s.plot(Xon, label='On')
            s.set_ylabel('{} / I'.format(t))
            s.set_xlabel('Channel')

        fig.suptitle(args.filename)
        plt.tight_layout()
        plt.show()




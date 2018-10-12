#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cuda.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <math.h>
#include <cmath>


/**
 *
 * Required inputs:
 * -- Nantennas total
 * -- Nchans
 * -- Coherent beam:
 *   -- antenna span (start, end)
 *   -- tscrunch
 *   -- fscrunch
 * -- Incoherent beam:
 *   -- antenna span (start, end)
 *   -- tscrunch
 *   -- fscrunch
 *
 */


/**
 *
 * How many threads needed?
 *  - Processing
 *  - Input
 *  - Output
 *  - Delays
 *
 */

void process(char2 const* taftp_voltages)
{

    // Read data from DADA buffer

    // Copy all data to GPU






    // Generate weights with padding (if not already generated)

    // Perform split transpose to extract data for coherent beamformer (with multiple-of-4 span)

    // Perform coherent beamforming using padded weights

    // Perform split transpose to extract data for incoherent beamformer (with multiple-of-2 span)

    // Perform incoherent beamforming

    // Copy coherent beam data to host

    // Copy incoherent beam data to host

    // Write data to DADA buffer

}
#include <math_constants.h>
__global__ int waveNumberMatrix(double* waveNumbers, double* horizontal,
        double* waveLengths, int beamNumber, int freqNumber);

__global__ int weightMatrix(double* weights, double* matrixLeft,
        double* matrixRight, int matrixLeftWidth, int matrixRightWidth, int freqChannels);

int readAntennaPosition(const char * fileName, double * positions);
int readBeamPosition(const char * fileName, double * positions);
int writeWeigts(const char * fileName, double * weights,
        int antennaNumber, int beamNumber, int freqChannels);
unsigned int countLineOfFile(const char * fileName);

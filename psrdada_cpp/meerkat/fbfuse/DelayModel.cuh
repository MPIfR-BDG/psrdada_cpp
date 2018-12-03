#ifndef PSRDADA_CPP_MEERKAT_DELAYMODEL_CUH
#define PSRDADA_CPP_MEERKAT_DELAYMODEL_CUH

// Note: This is a cuh file because float2 is a CUDA vector type.
//
// POD struct containing the layout of the shared memory
// buffer as written by the Python client
//
// Note: This cannot be pinned by the Nvidia driver as the memory is not
// a 4096 byte multiple (it is tiny though, so pinning should not be necessary)
struct DelayModel
{
    double epoch;
    double duration;
    float2 delays[FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS]; // Compile time constants
};

#endif //PSRDADA_CPP_MEERKAT_DELAYMODEL_CUH
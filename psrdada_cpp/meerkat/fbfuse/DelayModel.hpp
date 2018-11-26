#ifndef PSRDADA_CPP_MEERKAT_DELAYMODEL_HPP
#define PSRDADA_CPP_MEERKAT_DELAYMODEL_HPP

// POD struct containing the layout of the shared memory
// buffer as written by the Python client
struct DelayModel
{
    double epoch;
    double duration;
    float2 delays[FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS]; // Compile time constants
};

#endif //PSRDADA_CPP_MEERKAT_DELAYMODEL_HPP
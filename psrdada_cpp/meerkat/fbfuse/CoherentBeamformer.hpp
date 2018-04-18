#ifndef PSRDADA_CPP_MEERKAT_COHERENTBEAMFORMER_HPP
#define PSRDADA_CPP_MEERKAT_COHERENTBEAMFORMER_HPP


struct char4x2
{
    char4 x;
    char4 y;
};

struct char2x4
{
    char2 x;
    char2 y;
    char2 z;
    char2 w;
};

__forceinline__ __device__
void dp4a(int &c, const int &a, const int &b) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
  char4 &a4 = *((char4*)&a);
  char4 &b4 = *((char4*)&b);
  c += a4.x*b4.x;
  c += a4.y*b4.y;
  c += a4.z*b4.z;
  c += a4.w*b4.w;
#endif
}

__forceinline__ __device__
int2 int2_transpose(int2 const &input)
{
    char2x4 a;
    char4x2 b;
    a = (*(char2x4*)&input);
    b.x.x = a.x.x;
    b.x.y = a.y.x;
    b.x.z = a.z.x;
    b.x.w = a.w.x;
    b.y.x = a.x.y;
    b.y.y = a.y.y;
    b.y.z = a.z.y;
    b.y.w = a.w.y;
    return (*(int2*)&b);
}


class CoherentBeamformer
{
private:
    SubarrayConfiguration const& _subarray_config;
    BeamConfiguration const& _beam_config;
    WeightsCalculator _weights_calculator; //Needs subarray config and beam config
    voltage_t* _voltages; // Device buffer for storing voltages in FTPA order
    power_t* _powers; // Device buffer for storing beamformed powers in FBT order

public:
    CoherentBeamformer(SubarrayConfig const& subarray_config, BeamConfig const& beam_config);
    ~CoherentBeamformer();

    power_t const* form_beams(voltage_t const* voltages, power_t* _powers, time_t time);

};

#endif //PSRDADA_CPP_MEERKAT_COHERENTBEAMFORMER_HPP

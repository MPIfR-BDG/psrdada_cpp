#include "psrdada_cpp/transpose_client.hpp"
#include<boost/process.hpp>


{   //namespace psrdada_cpp 

namespace bp = boost::process;
int main()
{

   // Make a large dadabuffer
   {
       std::string key = "dada";
       boost::filesystem::path p = "/home/krajwade/libpsrdada/bin/dada_db";
       bp::child c0(p, "-k", key);
       co.wait();
   //Write Data into a dada buffer
       {
           std::vector<unsigned char> input_data(64*128*64*32*6);
           std::string logname = "Logger";
           Multilog log(logname);       
           DadaWriteClient writer0(string_to_key(key),log);
           in = input_data.data();
           auto stream = writer0.data_stream();
           auto out = stream.next();
           std::memcpy(out.ptr(),in,input_data.size());
           out.used_bytes(input_data.size()); 
           stream.release();
           writer0::~DadaWriteClient();
           std::string* keys[];
           *(keys) = "dadc";
           *(keys + 1) = "dade"
           *(keys + 2) = "dae0"
           *(keys + 3) = "dae2"
           *(keys + 4) = "dae4"
           *(keys + 5) = "dae6"
           
    // Running the psrdada to create 6 buffers 
           bp::child c1(p, "-k", *keys[0]);
           bp::child c2(p, "-k", *keys[1]);
           bp::child c3(p, "-k", *keys[2]);
           bp::child c4(p, "-k", *keys[3]);
           bp::child c5(p, "-k", *keys[4]);
           bp::child c6(p, "-k", *keys[5]);

           c1.wait();
           c2.wait();
           c3.wait();
           c4.wait();
           c5.wait();
           c6.wait();

    // Read to transpose 
           RawBytes& input;
           RawBytes& transposed, transposed1,transposed2,transposed3,transposed4,transposed5;   
           DadaReadClient  reader(,log);
           DadaWriteClient* writer1[];
           TransposeClient transpose(reader,writer1[],keys,6);
           input = transpose.read_to_transpose();
    
    // Do the transpose for all beams
           transposed = transpose.do_transpose(input,0);
           transposed1 = transpose.do_transpose(input,1);
           transposed2 = transpose.do_transpose(input,2);
           transposed3 = transpose.do_transpose(input,3);
           transposed4 = transpose.do_transpose(input,4);
           transposed5 = transpose.do_transpose(input,5);
           
 
    // Writing out each beam to 6 dada buffers as 6 threads

          
           std::thread t1(transpose.write_transpose(),std::ref(transposed),std::ref(*writer1[0]));
           std::thread t2(transpose.write_transpose(),std::ref(transposed1),std::ref(*writer1[1]));                 
           std::thread t3(transpose.write_transpose(),std::ref(transposed2),std::ref(*writer1[2]));
           std::thread t4(transpose.write_transpose(),std::ref(transposed3),std::ref(*writer1[3]));    
           std::thread t5(transpose.write_transpose(),std::ref(transposed4),std::ref(*writer1[4]));
           std::thread t6(transpose.write_transpose(),std::ref(transposed5),std::ref(*writer1[5]));

           t1.join();
           t2.join();
           t3.join();
           t4.join();
           t5.join();
           t6.join();
           
       }

    // Destroy the buffers
   }

}

}

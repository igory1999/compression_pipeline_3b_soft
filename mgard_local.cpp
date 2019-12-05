//#include <regex>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>
#include "mgard_local.h"
#include <timing.h>
#include <cstdio> //temporary fix until regex works


void MGARD_Init(const std::string & config_file, MGARD_PARAMETERS * parameters)
{
  std::ifstream in(config_file);
  std::string line;
  //  std::regex e1 ("tolerance");
  //  std::regex e2 (".*\\s*=\\s*(.+)");
  //  std::smatch m;
  /*
  while(std::getline(in, line))
    {
      if(std::regex_search(line, m, e1))
        {
          parameters->tolerance = std::stod(std::regex_replace(line, e2, "$1"));
        }
    }
  */

  parameters->tolerance = 1.e-7;  
  while(std::getline(in, line))
    {
      if(sscanf(line.c_str(), "tolerance=%lf", &parameters->tolerance)) break;
    }
  in.close();
  std::cout << "tolerance = " << parameters->tolerance << std::endl;
}


void MGARD_Compress_Decompress(double *indata, std::vector<std::size_t> &shape,
			       MGARD_PARAMETERS *params, MGARD_OUTPUT *out,
			       long *compressT, long* decompressT, std::string varname)
{
  MGARD_Init("mgard.config", params);
  std::size_t inN = shape[0] * shape[1] * shape[2];
  std::size_t inSize = inN * sizeof(double);

  {
    //MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:compress:" << varname << ":start " << params->iteration;
    timing(message.str(), *params->log);
  }

  auto start = std::chrono::steady_clock::now();    
  double *tmp = (double*)malloc(inSize);//is it really necessary?
  memcpy ( tmp, indata, inSize );
  int nfib = 1;
  double s = 0;

  out->compressed = (unsigned char*)mgard_compress(1, tmp, out->compressed_size, shape[2],
						   shape[1], shape[0], params->tolerance); // why &params->tolerance?
  auto end = std::chrono::steady_clock::now();  
  *compressT = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();  
  
  {
    //MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:compress:" << varname << ":end " << params->iteration;
    timing(message.str(), *params->log);
  }

  {
    //MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:decompress:" << varname << ":start " << params->iteration;
    timing(message.str(), *params->log);
  }


  start = std::chrono::steady_clock::now();  

  double quantizer=0.0; // It is unused?

  out->decompressed = (double*)mgard_decompress(1, quantizer, out->compressed, out->compressed_size,
						shape[2], shape[1], shape[0]);
  free(tmp);
  end = std::chrono::steady_clock::now();
  *decompressT = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();    

  {
    //MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:decompress:" << varname << ":end " << params->iteration;
    timing(message.str(), *params->log);
  }

}

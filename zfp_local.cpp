//#include <regex>
#include <fstream>
#include <cassert>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include "zfp_local.h"
#include <timing.h>
#include <cstdio>

void ZFP_Init(const std::string & config_file, ZFP_PARAMETERS * parameters)
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

void ZFP_Compress_Decompress(double *indata, std::vector<std::size_t> &shape,
			     ZFP_PARAMETERS *params, ZFP_OUTPUT *out,
			     long *compressT, long *decompressT, std::string varname)
{
  ZFP_Init("zfp.config", params);
  std::size_t insize = shape[0]*shape[1]*shape[2];
  zfp_type type = zfp_type_double;

  {
    //    MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:compress:" << varname << ":start " << params->iteration;
    timing(message.str(), *params->log);
  }

  auto start = std::chrono::steady_clock::now();  
  zfp_field *field = zfp_field_1d(indata, type, insize);
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_accuracy(zfp, params->tolerance);
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  out->compressed = malloc(bufsize);
  bitstream* stream = stream_open(out->compressed, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  out->compressed_size = zfp_compress(zfp, field);
  auto end = std::chrono::steady_clock::now();
  *compressT = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();  

  {
    //    MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:compress:" << varname << ":end " << params->iteration;
    timing(message.str(), *params->log);
  }

  {
    //    MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:decompress:" << varname << ":start " << params->iteration;
    timing(message.str(), *params->log);
  }

  start = std::chrono::steady_clock::now();    
  out->decompressed = malloc(insize*sizeof(double));
  zfp_field* field_dec = zfp_field_1d(out->decompressed, type, insize);
  zfp_stream_rewind(zfp);
  std::size_t size = zfp_decompress(zfp, field_dec);
  end = std::chrono::steady_clock::now();
  *decompressT = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();  

  {
    //    MPI_Barrier(*params->comm_ptr);
    std::ostringstream message;
    message << "compress:decompress:" << varname << ":end " << params->iteration;
    timing(message.str(), *params->log);
  }
}

#include "sz_local.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <timing.h>
#include <cstdio>

void SZ_Compress_Decompress(double *indata, std::vector<std::size_t> &shape,
                             SZ_PARAMETERS *params, SZ_OUTPUT *out,
                             long *compressT, long *decompressT, std::string varname)
{
  SZ_Init("sz.config");
  {
    std::ostringstream message;
    message << "compress:compress:" << varname << ":start " << params->iteration;
    timing(message.str(), *params->log);
  }

  auto startT = std::chrono::steady_clock::now();
  out->compressed = SZ_compress(SZ_DOUBLE, indata, &out->compressed_size,
		       0, 0, shape[2], shape[1], shape[0]);

  auto endT = std::chrono::steady_clock::now();
  *compressT = 
    std::chrono::duration_cast<std::chrono::nanoseconds>(endT - startT).count();

  {
    std::ostringstream message;
    message << "compress:compress:" << varname << ":end " << params->iteration;
    timing(message.str(), *params->log);
  }

  {
    std::ostringstream message;
    message << "compress:decompress:" << varname << ":start " << params->iteration;
    timing(message.str(), *params->log);
  }

  startT = std::chrono::steady_clock::now();      
  out->decompressed = (double*)SZ_decompress(SZ_DOUBLE, out->compressed, out->compressed_size,
				0, 0, shape[2], shape[1], shape[0]);
  endT = std::chrono::steady_clock::now();
  *decompressT = 
    std::chrono::duration_cast<std::chrono::nanoseconds>(endT - startT).count();

  {
    std::ostringstream message;
    message << "compress:decompress:" << varname << ":end " << params->iteration;
    timing(message.str(), *params->log);
  }
  SZ_Finalize();
}


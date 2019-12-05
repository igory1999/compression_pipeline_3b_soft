#include "mgard_api.h"
#include <fstream>
#include <string>
#include <mpi.h>  


struct MGARD_PARAMETERS
{
  double tolerance;
  int iteration;
  std::ofstream *log;
  MPI_Comm *comm_ptr;
};

struct MGARD_OUTPUT
{
  unsigned char *compressed = nullptr;
  double *decompressed = nullptr;
  int compressed_size;
};

void MGARD_Init(const std::string & config_file, MGARD_PARAMETERS * parameters);

void MGARD_Compress_Decompress(double *indata, std::vector<std::size_t> &shape,
			       MGARD_PARAMETERS *params, MGARD_OUTPUT *out,
			       long *compressT, long* decompressT, std::string varname);

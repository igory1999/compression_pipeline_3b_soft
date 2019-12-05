#include <sz.h>
#include <vector>
#include <string>
#include <mpi.h>

struct SZ_PARAMETERS
{
  double tolerance;
  int iteration;
  std::ofstream *log;
  MPI_Comm *comm_ptr;
};

struct SZ_OUTPUT
{
  unsigned char *compressed = nullptr;
  double *decompressed = nullptr;
  std::size_t compressed_size;
};


void SZ_Compress_Decompress(double *indata, std::vector<std::size_t> &shape,
			    SZ_PARAMETERS *params, SZ_OUTPUT *out,
			    long *compressT, long *decompressT, std::string varname);

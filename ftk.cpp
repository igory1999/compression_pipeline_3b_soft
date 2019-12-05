#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <mpi.h>
#include <adios2.h>
#include <timing.h>
#include "ftk_3D_interface.h"
#include "gaussian_filter.h"

using namespace GaussianFilter;

struct FTK_PARAMETERS
{
  int tile0;
  int tile1;
  int tile2;
  int window;
  double sigma;
  void print()
  {
    std::cout << "======" << std::endl;
    std::cout << "FTK_PARAMETERS" << std::endl;
    std::cout << "sigma = " << sigma << std::endl;
    std::cout << "window = " << window << std::endl;
    std::cout << "tile = (" << tile0 << ", " << tile1 << ", " << tile2 << ")" << std::endl; 
    std::cout << "======" << std::endl;
  }
};

void usage()
{
  std::cout << "mpirun -n 1 ftk_main <original data file> <lossy data file> <output file> <nthreads>" << std::endl;
}

void FTK_Init(const std::string & config_file, FTK_PARAMETERS * parameters)
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

  parameters->tile0 = 8;  
  parameters->tile1 = 8;  
  parameters->tile2 = 8;
  parameters->window = 3;
  parameters->sigma = 0.2;

  double tmp;
  while(std::getline(in, line))
    {
      if(sscanf(line.c_str(), "tile0=%lf", &tmp)) parameters->tile0 = static_cast<int>(tmp);
      if(sscanf(line.c_str(), "tile1=%lf", &tmp)) parameters->tile1 = static_cast<int>(tmp);
      if(sscanf(line.c_str(), "tile2=%lf", &tmp)) parameters->tile2 = static_cast<int>(tmp);
      if(sscanf(line.c_str(), "window=%lf", &tmp)) parameters->window = static_cast<int>(tmp);
      if(sscanf(line.c_str(), "sigma=%lf", &tmp)) parameters->sigma = tmp;
    }
  in.close();
  parameters->print();
}

void filter(double *d, int L, double sigma, ViewMatrixConstType gg, int tile0, int tile1, int tile2)
{
  Kokkos::View<double***, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_data (d, L, L, L);
  auto d_data = Kokkos::create_mirror_view_and_copy(Kokkos::Cuda(), h_data);
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::Cuda> data("data", L, L, L);
  Kokkos::deep_copy(data, d_data);
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::Cuda> result("result", L, L, L);  
  apply_kernel(data, result, gg, tile0, tile1, tile2);
  Kokkos::deep_copy(d_data, result);
  Kokkos::deep_copy(h_data, d_data);
}

int scan(int *sizes, int rank, int n, int *total)
{
  int subtotal = 0;
  for(int i=0;i<rank;++i)
      subtotal += sizes[i];
  *total = subtotal;
  for(int i = rank; i < n; ++i)
    *total += sizes[i];
  return subtotal;
}

void featurePut(std::vector<critical_point_t> & features, int offset, int total,
                adios2::Variable<double> & var_features_out, adios2::Engine & writer)
{
  int N = features.size();
  //  std::cout<<"In featurePut N = " << N << " total = " << total << " offset=" << offset << std::endl;
  const adios2::Dims start = {static_cast<long unsigned int>(offset), 0};
  const adios2::Dims count = {static_cast<long unsigned int>(N), 4};
  const adios2::Dims shape = {static_cast<long unsigned int>(total), 4};
  var_features_out.SetShape(shape);
  const adios2::Box<adios2::Dims> sel(start, count);
  var_features_out.SetSelection(sel);
  
  adios2::Variable<double>::Span features_span =
    writer.Put<double>(var_features_out);
  
  for(int i = 0, j = 0; i < N; ++i, j+=4)
    {
      features_span.at(j+0) = features[i].x[0];
      features_span.at(j+1) = features[i].x[1];
      features_span.at(j+2) = features[i].x[2];
      features_span.at(j+3) = features[i].v;        
    }
}


int main(int argc, char **argv)
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  Kokkos::initialize( argc, argv );
  int rank, comm_size, wrank;

  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  const unsigned int color = 2;
  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  std::ostringstream log_fn;
  log_fn << "ftk_" << rank << ".log";
  std::ofstream log(log_fn.str());

  
  {
    char processor_name[100];
    int name_len = 100;
    MPI_Get_processor_name(processor_name, &name_len);

    std::ostringstream message;
    message << "hostname=" << processor_name << std::endl;
    timing(message.str(), log);
  }


  if(comm_size != 4)
    {
      std::cout << "Currently only 4 MPI ranks are supported" << std::endl;
      usage();
      MPI_Finalize();
      return 1;
    }
  
  if(argc != 5)
    {
      if(!rank)
	usage();
      MPI_Finalize();
      return 1;

    }

  FTK_PARAMETERS ftk_parameters;
  FTK_Init("ftk.config", &ftk_parameters);

  std::string original_fn = argv[1];
  std::string lossy_fn = argv[2];
  std::string ftk_fn = argv[3];
  int nthreads = std::stoi(argv[4]);
  
  std::size_t u_global_size, v_global_size;
  std::size_t u_local_size, v_local_size;  
  std::vector<std::size_t> shape(3);
  std::vector<std::size_t> local_shape(3);
  bool firstStep = true;
    
  std::vector<double> u_original;
  std::vector<double> v_original;
  std::vector<double> u_lossy;
  std::vector<double> v_lossy;

  std::vector<double> u_original_big;
  std::vector<double> v_original_big;
  std::vector<double> u_lossy_big;
  std::vector<double> v_lossy_big;

  MPI_Datatype critical_point_type;
  int structlen = 2;
  int blocklengths[structlen]; 
  MPI_Datatype types[structlen];
  MPI_Aint displacements[structlen];

  struct critical_point_t myobject;
  
  blocklengths[0] = 3;
  types[0] = MPI_FLOAT;
  displacements[0] = (size_t)&(myobject.x[0]) - (size_t)&myobject;
  
  blocklengths[1] = 1;
  types[1] = MPI_DOUBLE;
  displacements[1] = (size_t)&(myobject.v) - (size_t)&myobject;
  
  MPI_Type_create_struct(structlen, blocklengths, 
                         displacements, types, &critical_point_type);
  MPI_Type_commit(&critical_point_type);

  adios2::Variable<double> var_u_original, var_v_original;
  adios2::Variable<int> var_step_original;
  adios2::Variable<double> var_u_lossy, var_v_lossy;
  adios2::Variable<int> var_step_lossy;  
  
  adios2::Variable<double> var_u_original_out, var_v_original_out;
  adios2::Variable<double> var_u_lossy_out, var_v_lossy_out;
  adios2::Variable<double> var_u_original_features_out, var_v_original_features_out;
  adios2::Variable<double> var_u_lossy_features_out, var_v_lossy_features_out;    
  adios2::Variable<int> var_u_original_features_n_out, var_u_lossy_features_n_out,
    var_v_original_features_n_out, var_v_lossy_features_n_out;
  
  adios2::Variable<int> var_u_distance_d_features_out, var_v_distance_d_features_out;
  adios2::Variable<double> var_u_distance_n_features_out, var_v_distance_n_features_out;    
   
  adios2::ADIOS ad ("adios2.xml", comm, adios2::DebugON);
  adios2::IO reader_original_io;
  adios2::IO reader_lossy_io;
  adios2::IO writer_ftk_io;
  
  adios2::Engine reader_original;
  adios2::Engine reader_lossy;
  adios2::Engine writer_ftk;

  reader_original_io = ad.DeclareIO("OriginalOutput");
  reader_lossy_io = ad.DeclareIO("DecompressedOutput");

  
  reader_original = reader_original_io.Open(original_fn,
					    adios2::Mode::Read, comm);
  reader_lossy = reader_lossy_io.Open(lossy_fn,
				      adios2::Mode::Read, comm);


  if(rank == 0)
    {
      writer_ftk_io = ad.DeclareIO("FTK");
      writer_ftk = writer_ftk_io.Open(ftk_fn,
				      adios2::Mode::Write, MPI_COMM_SELF);
    }

  int stepAnalysis = 0;
  while(true)
    {
      {
	std::ostringstream message;
	message << "ftk:read:original:start " << stepAnalysis;
	timing(message.str(), log);
      }
      
      adios2::StepStatus read_original_status = reader_original.BeginStep();
      if (read_original_status == adios2::StepStatus::NotReady)
	{
	  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	  continue;
	}
      else if (read_original_status != adios2::StepStatus::OK)
	{
	  break;
	}
      
      {
	log << "line=" <<__LINE__ << std::endl;
      }
      
      int step_original = reader_original.CurrentStep();
      var_u_original = reader_original_io.InquireVariable<double>("U/original");
      var_v_original = reader_original_io.InquireVariable<double>("V/original");
      var_step_original = reader_original_io.InquireVariable<int>("step");
      shape = var_u_original.Shape();
      
      {
	log << "line=" << __LINE__ << std::endl;
      }
      
      u_global_size = shape[0] * shape[1] * shape[2];
      u_local_size  = u_global_size/comm_size;
      //u_local_size = u_global_size;
      v_global_size = shape[0] * shape[1] * shape[2];
      v_local_size  = v_global_size/comm_size;
      //v_local_size  = v_global_size;
      
      size_t count1 = shape[0]/comm_size;
      //size_t count1 = shape[0];
      size_t start1 = count1 * rank;
      //size_t start1 = 0;

      if (rank == comm_size-1) {
	count1 = shape[0] - count1 * (comm_size - 1);
      }
      
      std::vector<std::size_t> local_shape = {count1, shape[1], shape[2]};
      //local_shape[0] = shape[0];
      //local_shape[1] = shape[1];
      //local_shape[2] = shape[2];
      
      var_u_original.SetSelection(adios2::Box<adios2::Dims>(
	{start1, 0, 0},
	{count1, shape[1], shape[2]}));
      var_v_original.SetSelection(adios2::Box<adios2::Dims>(
	{start1, 0, 0},
	{count1, shape[1], shape[2]}));
      
      
      {
	std::ostringstream message;
	message << "line="<<__LINE__ << std::endl;
      }
      
      reader_original.Get<double>(var_u_original, u_original);
      reader_original.Get<double>(var_v_original, v_original);
      reader_original.EndStep();
      
      {
	std::ostringstream message;
	message << "ftk:read:original:end " << stepAnalysis;
	timing(message.str(), log);
      }


      {
	std::ostringstream message;
	message << "ftk:gather:original:start " << stepAnalysis;
	timing(message.str(), log);
      }

      int bigsize = shape[0]*shape[1]*shape[2];

      if(rank == 0)
	u_original_big.resize(bigsize);
      MPI_Gather(u_original.data(), u_original.size(), MPI_DOUBLE,
		 u_original_big.data(), u_original.size(), MPI_DOUBLE,
		 0, comm);

      u_original.resize(0);

      if(rank == 2)
	v_original_big.resize(bigsize);
      MPI_Gather(v_original.data(), v_original.size(), MPI_DOUBLE,
		 v_original_big.data(), v_original.size(), MPI_DOUBLE,
		 2, comm);

      v_original.resize(0);

      {
	std::ostringstream message;
	message << "ftk:gather:original:end " << stepAnalysis;
	timing(message.str(), log);
      }   
      
      {
	std::ostringstream message;
	message << "ftk:read:lossy:start " << stepAnalysis;
	timing(message.str(), log);
      }
      
      adios2::StepStatus read_lossy_status = reader_lossy.BeginStep();
      if (read_lossy_status == adios2::StepStatus::NotReady)
	{
	  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	  continue;
	}
      else if (read_lossy_status != adios2::StepStatus::OK)
	{
	  break;
	}

      
      int step_lossy = reader_lossy.CurrentStep();
      var_u_lossy = reader_lossy_io.InquireVariable<double>("U/decompressed");
      var_v_lossy = reader_lossy_io.InquireVariable<double>("V/decompressed");
      var_step_lossy = reader_lossy_io.InquireVariable<int>("step");
      
      var_u_lossy.SetSelection(adios2::Box<adios2::Dims>(
	{start1, 0, 0},
	{count1, shape[1], shape[2]}));
      var_v_lossy.SetSelection(adios2::Box<adios2::Dims>(
	{start1, 0, 0},
	{count1, shape[1], shape[2]}));
      
      reader_lossy.Get<double>(var_u_lossy, u_lossy);
      reader_lossy.Get<double>(var_v_lossy, v_lossy);
      reader_lossy.EndStep();
      
      {
	std::ostringstream message;
	message << "ftk:read:lossy:end " << stepAnalysis;
	timing(message.str(), log);
      }


      {
	std::ostringstream message;
	message << "ftk:gather:lossy:start " << stepAnalysis;
	timing(message.str(), log);
      }

      if(rank == 1)
	u_lossy_big.resize(bigsize);
      MPI_Gather(u_lossy.data(), u_lossy.size(), MPI_DOUBLE,
		 u_lossy_big.data(), u_lossy.size(), MPI_DOUBLE,
		 1, comm);
      u_lossy.resize(0);

      if(rank == 3)
	v_lossy_big.resize(bigsize);
      MPI_Gather(v_lossy.data(), v_lossy.size(), MPI_DOUBLE,
		 v_lossy_big.data(), v_lossy.size(), MPI_DOUBLE,
		 3, comm);
      v_lossy.resize(0);

      {
	std::ostringstream message;
	message << "ftk:gather:lossy:end " << stepAnalysis;
	timing(message.str(), log);
      }   


      //FTK start

      int l = ftk_parameters.window;
      double sigma = ftk_parameters.sigma;

      ViewMatrixType g("gaussian", l, l, l);
      if(sigma > 0)
	generate_gaussian(sigma,  g);
      ViewMatrixConstType gg = g;
      int tile0=ftk_parameters.tile0, tile1=ftk_parameters.tile1, tile2=ftk_parameters.tile2;


      std::vector<critical_point_t> features_original_u;
      std::vector<critical_point_t> features_lossy_u;
      std::vector<critical_point_t> features_original_v;
      std::vector<critical_point_t> features_lossy_v;


      if(rank == 0)
	{

	  MPI_Request requests[6];
	  MPI_Status status[6];
	  int size[3];

	  {
	    std::ostringstream message;
	    message << "ftk:compute:start " << stepAnalysis;
	    timing(message.str(), log);
	  }
	  
	  if(sigma > 0)
	    filter(u_original_big.data(), shape[0], sigma, gg, tile0, tile1, tile2);
	  features_original_u =
	    extract_features(u_original_big.data(), shape[0], shape[1], shape[2], nthreads);


	  {
	    std::ostringstream message;
	    message << "ftk:compute:end " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  {
	    std::ostringstream message;
	    message << "ftk:receive:start " << stepAnalysis;
	    timing(message.str(), log);
	  }


	  MPI_Recv(&size[0], 1, MPI_INT, 1, stepAnalysis, comm, &status[0]);
	  features_lossy_u.resize(size[0]);
	  MPI_Recv(features_lossy_u.data(), size[0], critical_point_type, 1, stepAnalysis, comm, &status[1]);

	  MPI_Recv(&size[1], 1, MPI_INT, 2, stepAnalysis, comm, &status[2]);
	  features_original_v.resize(size[1]);
	  MPI_Recv(features_original_v.data(), size[1], critical_point_type, 2, stepAnalysis, comm, &status[3]);

	  MPI_Recv(&size[2], 1, MPI_INT, 3, stepAnalysis, comm, &status[4]);
	  features_lossy_v.resize(size[2]);
	  MPI_Recv(features_lossy_v.data(), size[2], critical_point_type, 3, stepAnalysis, comm, &status[5]);

	  {
	    std::ostringstream message;
	    message << "ftk:receive:end " << stepAnalysis;
	    timing(message.str(), log);
	  }

	}

      if(rank == 1)
	{
	  MPI_Status status;
	  int size;

	  {
	    std::ostringstream message;
	    message << "ftk:compute:start " << stepAnalysis;
	    timing(message.str(), log);
	    log << "u_lossy_size() = " << u_lossy.size() << std::endl;
	    log.flush();
	  }

	  if(sigma > 0)
	    filter(u_lossy_big.data(), shape[0], sigma, gg, tile0, tile1, tile2);
	  features_lossy_u =
	    extract_features(u_lossy_big.data(), shape[0], shape[1], shape[2], nthreads);

	  {
	    std::ostringstream message;
	    message << "ftk:compute:end " << stepAnalysis;
	    timing(message.str(), log);
	  }


	  {
	    std::ostringstream message;
	    message << "ftk:send:start " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  size = features_lossy_u.size();
	  MPI_Send(&size, 1, MPI_INT, 0, stepAnalysis, comm);
	  MPI_Send(features_lossy_u.data(), size, critical_point_type, 0, stepAnalysis, comm);

	  {
	    std::ostringstream message;
	    message << "ftk:send:end " << stepAnalysis;
	    timing(message.str(), log);
	  }
	}

      if(rank == 2)
	{
	  MPI_Status status;
	  int size;

	  {
	    std::ostringstream message;
	    message << "ftk:compute:start " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  if(sigma > 0) 
	    filter(v_original_big.data(), shape[0], sigma, gg, tile0, tile1, tile2);
	  features_original_v =
	    extract_features(v_original_big.data(), shape[0], shape[1], shape[2], nthreads);


	  {
	    std::ostringstream message;
	    message << "ftk:compute:end " << stepAnalysis;
	    timing(message.str(), log);
	  }


	  {
	    std::ostringstream message;
	    message << "ftk:send:start " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  size = features_original_v.size();
	  MPI_Send(&size, 1, MPI_INT, 0, stepAnalysis, comm);
	  MPI_Send(features_original_v.data(), size, critical_point_type, 0, stepAnalysis, comm);

	  {
	    std::ostringstream message;
	    message << "ftk:send:end " << stepAnalysis;
	    timing(message.str(), log);
	  }
	}

      if(rank == 3)
	{

	  MPI_Status status;
	  int size;

	  {
	    std::ostringstream message;
	    message << "ftk:compute:start " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  if(sigma > 0)
	    filter(v_lossy_big.data(), shape[0], sigma, gg, tile0, tile1, tile2);
	  features_lossy_v =
	    extract_features(v_lossy_big.data(), shape[0], shape[1], shape[2], nthreads);

	  {
	    std::ostringstream message;
	    message << "ftk:compute:end " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  {
	    std::ostringstream message;
	    message << "ftk:send:start " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  size = features_lossy_v.size();
	  MPI_Send(&size, 1, MPI_INT, 0, stepAnalysis, comm);
	  MPI_Send(features_lossy_v.data(), size, critical_point_type, 0, stepAnalysis, comm);

	  {
	    std::ostringstream message;
	    message << "ftk:send:end " << stepAnalysis;
	    timing(message.str(), log);
	  }

	}


      if(rank == 0)
	{

	  if (firstStep)
	    {
	      {
		log << "line="<<__LINE__ << std::endl;
	      }

	      var_u_original_features_out =
		writer_ftk_io.DefineVariable<double> ("U_features/original",
		  { 1, 4},
		  { 0, 0},
		  { 1, 4} );
	      var_u_original_features_n_out =
		writer_ftk_io.DefineVariable<int>("U_features_n/original", {adios2::LocalValueDim});
	      var_v_original_features_out =
		writer_ftk_io.DefineVariable<double> ("V_features/original",
		  { 1, 4},
		  { 0, 0},
		  { 1, 4} );
	      var_v_original_features_n_out =
		writer_ftk_io.DefineVariable<int>("V_features_n/original", {adios2::LocalValueDim});      
	      var_u_lossy_features_out =
		writer_ftk_io.DefineVariable<double> ("U_features/lossy",
		  { 1, 4},
		  { 0, 0},
		  { 1, 4} );
	      var_u_lossy_features_n_out =
		writer_ftk_io.DefineVariable<int>("U_features_n/lossy", {adios2::LocalValueDim});         
	      var_v_lossy_features_out =
		writer_ftk_io.DefineVariable<double> ("V_features/lossy",
		  { 1, 4},
		  { 0, 0},
		  { 1, 4} );
	      var_v_lossy_features_n_out =
		writer_ftk_io.DefineVariable<int>("V_features_n/lossy", {adios2::LocalValueDim});
	      
	      var_u_distance_d_features_out =
		writer_ftk_io.DefineVariable<int>("U_features_distance/difference", {adios2::LocalValueDim});
	      var_u_distance_n_features_out =
		writer_ftk_io.DefineVariable<double>("U_features_distance/normalized", {adios2::LocalValueDim});
	      var_v_distance_d_features_out =
		writer_ftk_io.DefineVariable<int>("V_features_distance/difference", {adios2::LocalValueDim});
	      var_v_distance_n_features_out =
		writer_ftk_io.DefineVariable<double>("V_features_distance/normalized", {adios2::LocalValueDim});            
	      firstStep = false;
	    }

	  {
	    log << "line="<<__LINE__ << std::endl;
	  }


	  int distance_u_diff, distance_v_diff;
	  double distance_u_norm, distance_v_norm;
	  
	  distance_between_features(features_original_u, features_lossy_u,
				    &distance_u_diff, &distance_u_norm);
	  distance_between_features(features_original_v, features_lossy_v,
				    &distance_v_diff, &distance_v_norm);
	  

	  {
	    log << "line="<<__LINE__ << std::endl;
	  }

	  const int nuo = features_original_u.size();
	  const int nul = features_lossy_u.size();
	  const int nvo = features_original_v.size();
	  const int nvl = features_lossy_v.size();
      
	  int nuo_n = nuo, nuo_offset = 0;
	  int nul_n = nul, nul_offset = 0;
	  int nvo_n = nvo, nvo_offset = 0;
	  int nvl_n = nvl, nvl_offset = 0;      

	  {
	    log << "nm " << nuo << " " << nul << " " << nvo << " " << nvl << std::endl;
	  }

	  {
	    std::ostringstream message;
	    message << "ftk:write:features:start " << stepAnalysis;
	    timing(message.str(), log);
	  }

	  writer_ftk.BeginStep ();
	  writer_ftk.Put<int>(var_u_original_features_n_out, &nuo);
	  writer_ftk.Put<int>(var_v_original_features_n_out, &nvo);
	  writer_ftk.Put<int>(var_u_lossy_features_n_out, &nul);
	  writer_ftk.Put<int>(var_v_lossy_features_n_out, &nvl);
	  
	  writer_ftk.Put<int>(var_u_distance_d_features_out, distance_u_diff);
	  writer_ftk.Put<int>(var_v_distance_d_features_out, distance_v_diff);
	  writer_ftk.Put<double>(var_u_distance_n_features_out, distance_u_norm);
	  writer_ftk.Put<double>(var_v_distance_n_features_out, distance_v_norm);
	  
	  featurePut(features_original_u, nuo_offset, nuo_n,
		     var_u_original_features_out, writer_ftk);
	  featurePut(features_original_v, nvo_offset, nvo_n,
		     var_v_original_features_out, writer_ftk);
	  featurePut(features_lossy_u, nul_offset, nul_n,
		     var_u_lossy_features_out, writer_ftk);
	  featurePut(features_lossy_v, nvl_offset, nvl_n,
		     var_v_lossy_features_out, writer_ftk);
	  
	  writer_ftk.EndStep ();
	  
	  {
	    std::ostringstream message;
	    message << "ftk:write:features:end " << stepAnalysis;
	    timing(message.str(), log);
	  }

	}
      ++stepAnalysis;
      MPI_Barrier(comm);
    }

  log.close();
  reader_original.Close();
  reader_lossy.Close();

  if(rank == 0)
    {
      writer_ftk.Close();
    }

  Kokkos::finalize();
  MPI_Type_free(&critical_point_type);
  MPI_Finalize();
  return 0;
}

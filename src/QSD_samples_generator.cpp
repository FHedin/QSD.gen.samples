/**
 * \file QSD_samples_generator.cpp
 * \copyright The 3-clause BSD license is applied to this software.
 * \author Florent Hédin
 * \author Tony Lelièvre
 * \author École des Ponts - ParisTech
 * \date 2016-2018
 */

#include <utility>
#include <limits>
#include <array>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>

#include "GelmanRubin.hpp"

#include "logger.hpp"
#include "rand.hpp"

#include "QSD_samples_generator.hpp"

using namespace std;

QSD_samples_generator::QSD_samples_generator(DATA& _dat,
                                             unique_ptr<ATOM[]>& _at,
                                             unique_ptr<MD_interface>& _md,
                                             unique_ptr<luaInterface>& _luaItf
                                            ) : dat(_dat),at(_at),md(_md),luaItf(_luaItf),
                                                params(luaItf->get_parsed_parameters_map()),
                                                lua_state_init(luaItf->get_function_state_init()),
                                                lua_check_state(luaItf->get_function_check_state()),
                                                lua_save_conditions(luaItf->get_function_save_conditions()),
                                                gr_functions(luaItf->get_gr_functions()),
                                                gr_names(luaItf->get_gr_names())
{
  MPI_setup();
  
  gr_check       = (uint32_t) stoul(params.at("checkGR"));
  fv_check       = (uint32_t) stoul(params.at("checkFV"));
  grTol          = stod(params.at("GRtol"));

  // assign a Fleming-Viot role to this replica, based on MPI rank
  my_FV_role = (i_am_master) ? REF_WALKER : FV_WALKER;

}

QSD_samples_generator::~QSD_samples_generator()
{
  MPI_clean();
}

void QSD_samples_generator::run()
{
  fprintf(stdout,"\nRunning a Fleming-Viot particle process.\n"
                 "Role of this replica is: %s\n",role_string.at(my_FV_role).c_str());

  // it is time to initialise the Lua code defining a state
  lua_state_init();
  
  uint32_t num_fv_cycles = 0;
  const uint32_t stop_after_N_fv_cycles = (uint32_t) dat.nsteps;
  /*
   * NOTE main loop here
   */
  do
  {
    LOG_PRINT(LOG_INFO,"New FV loop iteration...\n");
    
    fprintf(stdout,"\n//---------------------------------------------------------------------------------------------//\n");
    fprintf(stdout,    "New FV loop iteration...\n");
    
    fv_local_time = 0.;
    fv_e = ENERGIES();
    
    /*
     * Stage 1 (FV) : equivalent of decorrelation + dephasing done together
     *  using the FV approach and the GR analysis
     */
    do_FlemingViot_procedure();
    // there was already a barrier at the end of do_FlemingViot_procedure() so from here we suppose synchronization of all the replicas
    
    num_fv_cycles++;
    
    // update lua interface
    luaItf->set_lua_variable("epot",fv_e.epot());
    luaItf->set_lua_variable("ekin",fv_e.ekin());
    luaItf->set_lua_variable("etot",fv_e.etot());
    
    // call Lua code here for saving the N initial conditions obtained from the F-V procedure
    lua_save_conditions(fv_local_time);
    
    MPI_Barrier(global_comm);
    
    // use one of the N initial condition obtained in the current run as the starting point of the next iteration
    int32_t candidate = -1;
    if(my_FV_role == REF_WALKER)
      candidate = get_int32_min_max(0,mpi_gcomm_size-1);
    
    MPI_Bcast(&candidate,1,MPI_INT32_T,masterRank,global_comm);
    
    LOG_PRINT(LOG_INFO,"For the next FV iteration, initial coordinates will be the one currently associated to rank %d\n",candidate);
    if(LOG_SEVERITY >= LOG_INFO)
      fprintf(stdout,  "For the next FV iteration, initial coordinates will be the one currently associated to rank %d\n",candidate);
    
    if(mpi_id_in_gcomm == candidate)
      md->getState(nullptr,&fv_e,nullptr,at.get());
    
    MPIutils::mpi_broadcast_atom_array(dat,at.get(),candidate,global_comm);

    // we reset the omm time
    md->setSimClockTime(0.);
    
    if(mpi_id_in_gcomm != candidate)
    {
      // each rank uses the coordintes of the candidate
      md->setCrdsVels(at.get());
    }
    
    // now all ranks need to re-initialize the state identification code on Lua's side
    lua_state_init();
    
    /*
     * Ready to loop again
     */
    
    MPI_Barrier(global_comm);
    
  }while(num_fv_cycles < stop_after_N_fv_cycles);
  // NOTE End of main loop here

  MPI_Barrier(global_comm);
      
} // end of function run

void QSD_samples_generator::do_FlemingViot_procedure()
{
  LOG_PRINT(LOG_INFO,"Rank %d performing Fleming-Viot procedure\n",  mpi_id_in_gcomm);
  fprintf(stdout,    "Rank %d performing Fleming-Viot procedure\n",mpi_id_in_gcomm);
  
  /**
   *  the C++ interface for collecting Gelman-Rubin statistics
   *  For the moment only rank masterRank will manage it.
   *  The others regularly send their data (stored in gr_observations) to rank masterRank
   */
  std::unique_ptr<GelmanRubinAnalysis> gr = nullptr;
  
  // The ref walker is in charge of allocating the GelmanRubinAnalysis object, it will also check the convergence
  if(my_FV_role == REF_WALKER)
  {
    gr = unique_ptr<GelmanRubinAnalysis>(new GelmanRubinAnalysis((uint32_t)mpi_gcomm_size));
    // register grObservables parsed from lua file
    for(const pair<GR_function_name,GR_function>& p : gr_functions)
    {
      gr->registerObservable(Observable(p.first),grTol);
    }
  }
  
  // number of observables that the workers send to the ref walker every fv_check steps
  const uint32_t numObs = fv_check/gr_check;

  // This will store the observables sent by the workers to the ref walker
  map<GR_function_name,double*> recvObsMap;
  if(my_FV_role == REF_WALKER)
  {
    for(const GR_function_name& n: gr_names)
      recvObsMap[n] = new double[mpi_gcomm_size*numObs];
  }
  
  // stores requests used when gathering in non blocking mode on ref walker the observables sent by the workers
  vector<MPI_Request> igather_obs_reqs(gr_names.size(),MPI_REQUEST_NULL);
  
  // if this rank is requiring branching
  bool i_need_branching = false;

  // declaration and create of sattic memory windows for one sided MPI comms
  MPI_Win at_window        = MPI_WIN_NULL;
  MPI_Win pbc_window       = MPI_WIN_NULL;
  MPI_Win branching_window = MPI_WIN_NULL;

  // set the at[] as a window of memory accessible by other ranks: one sided communications
  MPI_Win_create(at.get(),dat.natom*sizeof(ATOM),1,      //void *base, MPI_Aint size, int disp_unit
                 rma_info,global_comm,&at_window);  // MPI_Info info, MPI_Comm comm, MPI_Win *win
  // same for the pbc
  MPI_Win_create(dat.pbc.data(),sizeof(dat.pbc),1,
                 rma_info,global_comm,&pbc_window);
  // and also for the i_need_branching variable
  MPI_Win_create(&i_need_branching,sizeof(bool),sizeof(bool),
                 rma_info,global_comm,&branching_window);

  // we need a window for exchanging Gelman-Rubin data but its size is varying so we create it here using a dynamic MPI window
  map<GR_function_name,MPI_Win> gr_windows;
  for(const GR_function_name& n: gr_names)
  {
    gr_windows[n] = MPI_WIN_NULL;
    MPI_Win_create_dynamic(rma_info,
                           global_comm,
                           &gr_windows[n]);
  }
  
  /*
   * Each mpi rank uses a map for storing observations of each observable during simulation
   * 
   * For each observable with a given GR_function_name, a vector of double stores the associated observations
   * 
   * From time to time this data is sent to masterRank which owns the GelmanRubinAnalysis object, and the masterRank checks convergence
   */
  map<GR_function_name,double*> gr_observations;
  
  /*
   * In order to use MPI_Win_create_dynamic we also need a record of the MPI_Aint 
   * address at which the vector<double>() are stored, and a static window for allowing RMA access to each of them
   */
  map<GR_function_name,MPI_Aint> gr_addr;
  map<GR_function_name,MPI_Win>  gr_windows_addr;
  
  for(const GR_function_name& n: gr_names)
  {
    //gr_observations[n] = vector<double>();
    gr_observations[n] = new double[numObs];
    gr_addr[n] = 0;
    gr_windows_addr[n] = MPI_WIN_NULL;
    MPI_Win_create(&(gr_addr[n]),sizeof(MPI_Aint),sizeof(MPI_Aint),
                   rma_info,global_comm,&(gr_windows_addr[n]));
  }

  md->getState(nullptr,&fv_e,nullptr,at.get());
  luaItf->set_lua_variable("epot",fv_e.epot());
  luaItf->set_lua_variable("ekin",fv_e.ekin());
  luaItf->set_lua_variable("etot",fv_e.etot());
//   luaItf->set_lua_variable("referenceTime",ref_clock_time);

  ////////////////////////////////////////////////////////////////////////////
  /*
   * loop until there is a converged distribution of FV walkers (i.e. ranks) within the state
   */
  uint32_t steps_since_last_check_state = 0;
  uint32_t steps_since_beginning = 0;
  uint32_t obsIndex = 0;
  
  bool converged = false;
//   bool reset_loop = false;
  
  MPI_Win converged_window = MPI_WIN_NULL;
//   MPI_Win reset_window = MPI_WIN_NULL;
  
  MPI_Win_create(&converged,sizeof(bool),sizeof(bool),
                 rma_info,global_comm,&converged_window);
//   MPI_Win_create(&reset_loop,sizeof(bool),sizeof(bool),
//                  rma_info,global_comm,&reset_window);
  
  /*
   * we will use when possible a non blocking barrier (ibarrier), and before checking the request associated to it,
   * we will try to perform independent computations in order to hide latency of the ibarrier
   */
  MPI_Request barrier_req = MPI_REQUEST_NULL;
  
  while(true)
  {

    md->doNsteps(gr_check);

    // wait until data sent in the igather (observables) arrived to the master node
    MPI_Waitall(gr_names.size(),igather_obs_reqs.data(),MPI_STATUSES_IGNORE);

    if(steps_since_last_check_state == 0)
    {
      obsIndex = 0;
    }
    
    /*
     * wait for the last ibarrier at the end of the previous loop ieration completed
     * if first iteration or if there was a loop iteration without barrier this has no effect
     */
    MPI_Wait(&barrier_req,MPI_STATUS_IGNORE);

    if(converged)
      break;
    
    steps_since_last_check_state += gr_check;
    steps_since_beginning += gr_check;
    
    //increment local time and continue
    fv_local_time += (double) gr_check * dat.timestep;

    md->getState(nullptr,&fv_e,nullptr,at.get());
    luaItf->set_lua_variable("epot",fv_e.epot());
    luaItf->set_lua_variable("ekin",fv_e.ekin());
    luaItf->set_lua_variable("etot",fv_e.etot());
    
    for(const pair<GR_function_name,GR_function>& p : gr_functions)
    {
      // first retrieve the type of observable (name), the pointer to the lua function (f) corresponding to it, and the vector of observations (v) corresponding to name 
      const GR_function_name& name = p.first;
      const GR_function&         f = p.second;
      
      gr_observations[name][obsIndex] = f();
    }
    obsIndex += 1;
    
    // do the following F-V branching procedure less often (every fv_check steps)
    //  than the GR accumulation phase (gr_check)
    if( steps_since_last_check_state%fv_check != 0 )
      continue;
    

    left_state = lua_check_state();
    i_need_branching = left_state;
    steps_since_last_check_state = 0;

    /*
     * we use a non-blocking barrier because we execute between the barrier and the corresponding MPI_Wait routine
     * some independent computations
     */
    MPI_Ibarrier(global_comm,&barrier_req);
    
    /*
     * attach the windows to the observable values
     */
    for(const GR_function_name& n: gr_names)
    {
      MPI_Win& w = gr_windows[n];
      MPI_Aint& addr = gr_addr[n];
      
      MPI_Win_attach(w, gr_observations[n], obsIndex*sizeof(double));
      MPI_Get_address(gr_observations[n],&addr);
    }
    
    MPI_Wait(&barrier_req,MPI_STATUS_IGNORE);
    
    /*
     *  F-V branching procedure will start here
     *  We should randomly choose one of the other ranks, and ask for its values for :
     *    + coordinates
     *    + velocities
     *    + PBCs
     *    + GR observables accumulation history
     * 
     *  Then update local ones with those values, and run again
     */

    // if required, find a valid candidate for branching
    if(i_need_branching)
    {
      // when a rank will branch it will send a communication request to another node (a 'candidate')
      int32_t candidate = -1;
      
      /*
       * A good candidate is :
       *  + another chain ...
       *  + ... not requiring branching
       */
      do
      {
        candidate = get_int32_min_max(0,mpi_gcomm_size-1);
        
        if(candidate == mpi_id_in_gcomm)
          continue;
        
        bool candidate_also_branching = false;
        
        MPI_Win_lock(MPI_LOCK_SHARED,candidate,0,branching_window);
        MPI_Get(&candidate_also_branching,1,   // void *origin_addr, int origin_count,
                MPI_CXX_BOOL,candidate,        // MPI_Datatype origin_datatype, int target_rank,
                0,1,                           // MPI_Aint target_disp, int target_count,
                MPI_CXX_BOOL,branching_window);// MPI_Datatype target_datatype, MPI_Win win
        MPI_Win_unlock(candidate,branching_window);
        
        if(!candidate_also_branching)
          break;
        
      }while(true);

      LOG_PRINT(LOG_DEBUG,"Rank %d left the ref state, initiating a F-V branching from candidate %d...\n",mpi_id_in_gcomm,candidate);
      
      // other ranks are allowed to read from the same candidate ( use of MPI_LOCK_SHARED )
      MPI_Win_lock(MPI_LOCK_SHARED,candidate,0,at_window);
      MPI_Win_lock(MPI_LOCK_SHARED,candidate,0,pbc_window);
      
      MPI_Get(at.get(),dat.natom*sizeof(ATOM), // void *origin_addr, int origin_count,
              MPI_BYTE,candidate,              // MPI_Datatype origin_datatype, int target_rank,
              0,dat.natom*sizeof(ATOM),        // MPI_Aint target_disp, int target_count,
              MPI_BYTE,at_window);             // MPI_Datatype target_datatype, MPI_Win win
      
      MPI_Get(dat.pbc.data(),sizeof(dat.pbc),  // void *origin_addr, int origin_count,
              MPI_BYTE,candidate,              // MPI_Datatype origin_datatype, int target_rank,   
              0,sizeof(dat.pbc),               // MPI_Aint target_disp, int target_count,
              MPI_BYTE,pbc_window);            // MPI_Datatype target_datatype, MPI_Win win

      // remove lock and proceed
      MPI_Win_unlock(candidate,at_window);
      MPI_Win_unlock(candidate,pbc_window);

      // do the same for gr data
      for(const GR_function_name& n: gr_names)
      {
        // step 1 : retrieve mem address for using RMA op on a dynamic window below
        MPI_Aint addr = 0;
        MPI_Win& gr_win_addr = gr_windows_addr[n];
        MPI_Win_lock(MPI_LOCK_SHARED,candidate,0,gr_win_addr);
        MPI_Get(&addr,1,MPI_AINT,candidate,
                0,1,MPI_AINT,gr_win_addr);
        MPI_Win_unlock(candidate,gr_win_addr);
        
        // step 2 : data exchange
        MPI_Win& gr_win = gr_windows[n];
        
        MPI_Win_lock(MPI_LOCK_SHARED,candidate,0,gr_win);
        MPI_Get(gr_observations[n],obsIndex,// void *origin_addr, int origin_count,
                MPI_DOUBLE,candidate, // MPI_Datatype origin_datatype, int target_rank,
                addr,obsIndex,         // MPI_Aint target_disp, int target_count,
                MPI_DOUBLE,gr_win);   // MPI_Datatype target_datatype, MPI_Win win
        MPI_Win_unlock(candidate,gr_win);
      }

      LOG_PRINT(LOG_DEBUG,"Rank %d properly branched data from rank %d\n",mpi_id_in_gcomm,candidate);
      LOG_PRINT(LOG_DEBUG,"Rank %d also branched GR observations from rank %d\n",mpi_id_in_gcomm,candidate);
      
      md->setCrdsVels(at.get());
      
      md->getState(nullptr,&fv_e,nullptr,nullptr);
      luaItf->set_lua_variable("epot",fv_e.epot());
      luaItf->set_lua_variable("ekin",fv_e.ekin());
      luaItf->set_lua_variable("etot",fv_e.etot());
      
      i_need_branching=false;
      candidate=-1;
    }

    // again this barrier can't be avoided otherwise we will detach memory too early
    // NOTE but we can still do something else before checking the barrier completion ...
    MPI_Ibarrier(global_comm,&barrier_req);
    
    // now we check barrier completion here
    MPI_Wait(&barrier_req,MPI_STATUS_IGNORE);
    
    // detach the windows now that the ibarrier has been checked 
    for(const GR_function_name& n: gr_names)
    {
      MPI_Win& w = gr_windows[n];
      MPI_Aint& addr = gr_addr[n];
      MPI_Win_detach(w, gr_observations[n]);
      addr = 0;
    }
    
    // then each rank sends its data to the REF_WALKER
    for(size_t i=0; i<gr_names.size(); i++)
    {
      const string& n = gr_names[i];
      
      double* recvObs = (my_FV_role == REF_WALKER) ? recvObsMap[n] : nullptr;
      
      MPI_Igather(gr_observations[n],numObs,MPI_DOUBLE,
                  recvObs,numObs,MPI_DOUBLE,
                  masterRank,global_comm,&igather_obs_reqs[i]);
    }

    /*
     * update GR statistics : check if each observable has converged
     * if convergence notify the workers by setting the corresponding boolean flag to true
     * the workers do not wait for this, they already looped to the beginning of the while loop and will start doing md again
     * after doing gr_check steps they will wait for the ibarrier at the end of this block to be completed;
     * this way we hide the latency time required for notifying the walkers behind the md dynamics time
     */
    if(my_FV_role == REF_WALKER)
    {
      MPI_Waitall(gr_names.size(),igather_obs_reqs.data(),MPI_STATUSES_IGNORE);
      
      for(const GR_function_name& n : gr_names)
      {
        double* recvObs = recvObsMap[n]; //.get();
        for(int32_t i=0; i<mpi_gcomm_size; i++)
        {
          const size_t from = i*numObs;
          gr->addNObservations(n,recvObs+from,numObs,i);
        }
      }
      
      gr->updateStatistics();

      gr->describe();
      if(LOG_SEVERITY == LOG_DEBUG)
      {
        for(uint32_t n=0; n<(uint32_t)mpi_gcomm_size; n++)
          gr->describeChain(n);
      }

      vector<bool> convergedVec = gr->check_convergence_all();
      
      bool lconverged = convergedVec[0];
      for(size_t n=1; n<gr_names.size(); n++)
        lconverged &= convergedVec[n];
      
      if(lconverged)
      {
        converged = true;
        MPI_Win_lock_all(MPI_MODE_NOCHECK,converged_window);
        for(int32_t i=1; i<mpi_gcomm_size; i++)
        {
          MPI_Put(&converged, //const void *origin_addr
                  1,MPI_CXX_BOOL, //int origin_count, MPI_Datatype origin_datatype,
                  i,0,  // int target_rank, MPI_Aint target_disp,
                  1,MPI_CXX_BOOL, // int target_count, MPI_Datatype target_datatype,
                  converged_window); // MPI_Win win
        }
        MPI_Win_unlock_all(converged_window);
      }
      
    }

    /*
     * again barrier can't be avoided but we hide it using a non blocking variant
     * the completion check is performed at the beginning of the while loop, AFTER a call to md->doNsteps(gr_check) in order to hide letency behind a small
     * amount of computations, ie md integration of gr_check steps
     */
    MPI_Ibarrier(global_comm,&barrier_req);
    
  }// convergence while loop
  
  LOG_PRINT(LOG_INFO,"Fleming-Viot converged after %.2lf ps\n",fv_local_time);
  fprintf(    stdout,"Fleming-Viot converged after %.2lf ps\n",fv_local_time);
  
  if(my_FV_role == REF_WALKER)
  {
    LOG_PRINT(LOG_INFO,"Fleming-Viot converged after %.2lf ps : Gelman-Rubin statistics are : \n",fv_local_time);
    
    gr->describe();
    if(LOG_SEVERITY == LOG_DEBUG)
    {
      for(uint32_t n=0; n<(uint32_t)mpi_gcomm_size; n++)
        gr->describeChain(n);
    }
    
    for(const GR_function_name& n : gr_names)
      delete[] recvObsMap[n];

    recvObsMap.clear();
  }

  //----------------------
  
  // reset GR object before continuing
  if(my_FV_role == REF_WALKER)
    gr->reset_all_chains();
  
  for(const GR_function_name& n: gr_names)
    delete[] gr_observations[n];
  
  gr_observations.clear();
  
  for(size_t i=0; i<gr_names.size(); i++)
  {
    if(igather_obs_reqs[i] != MPI_REQUEST_NULL)
      MPI_Request_free(&igather_obs_reqs[i]);
  }
  
  MPI_Win_free(&at_window);
  MPI_Win_free(&pbc_window);
  MPI_Win_free(&branching_window);
//   MPI_Win_free(&reset_window);
  MPI_Win_free(&converged_window);
  
  for(const GR_function_name& n: gr_names)
  {
    MPI_Win_free(&(gr_windows[n]));
    MPI_Win_free(&(gr_windows_addr[n]));
  }
  
} // end of do_FlemingViot_procedure

void QSD_samples_generator::MPI_setup()
{
  LOG_PRINT(LOG_DEBUG,"Call of the default MPI_setup() defined in file %s at line %d \n",__FILE__,__LINE__);
  
  // first a copy of the original MPI_COMM_WORLD is done
  MPI_Comm_dup(MPI_COMM_WORLD,&global_comm);
  // extract the original group handle
  MPI_Comm_group(global_comm, &global_group);
  
  MPI_Comm_rank(global_comm, &mpi_id_in_gcomm);
  MPI_Comm_size(global_comm, &mpi_gcomm_size);
  
  masterRank = 0;
  i_am_master = (mpi_id_in_gcomm == masterRank);
  
  // set some optimization flags for MPI_Win RMA operations
  MPI_Info_create(&rma_info);
  MPI_Info_set(rma_info,"same_size","true");
  MPI_Info_set(rma_info,"same_disp_unit","true");
  
}

void QSD_samples_generator::MPI_clean()
{
  LOG_PRINT(LOG_DEBUG,"Call of the default MPI_clean() defined in file %s at line %d \n",__FILE__,__LINE__);
  
  MPI_Group_free(&global_group);
  MPI_Comm_free(&global_comm);
  MPI_Info_free(&rma_info);
}

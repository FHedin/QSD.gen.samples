-- Input Lua script providing simulation parameters and functions for the QSD.gen.samples software
-- All lines starting with '--' are comments and thus ignored

-- Alanine dipeptide in gas phase, charmm FF

-- print("Lua version : ",_VERSION)

--------------------------------------------------------------------------------------------------------------
-- --------------- SIMULATION PARAMETERS ------------------------------
--------------------------------------------------------------------------------------------------------------

-- OpenMM platform to use
--  AUTO : let OpenMM find the fastest platform (default)
--  REF  : on cpu, not optimised, not parallellised : slow !
--  CPU  : on cpu, optimised, parallellised with threads
--  OCL  : on cpu or gpu of any brand, or any accelerating device available
--  CUDA : on nvidia gpu only, usually the fastest
-- OMMplatform = "AUTO"
-- OMMplatform = "REF"
OMMplatform = "CPU"
-- OMMplatform = "OCL"
-- OMMplatform = "CUDA"

-- load the integrator parameters from a serialised OpenMM XML file ;
-- no default, error if undefined
--  adapt the path to the file in the following line
integrator = { xml = "./mol/ala2vac/Integrator.xml" }

-- load the OpenMM System from a serialised XML file
-- no default, error if undefined
system = { xml = "./mol/ala2vac/System.xml" }

-- load the OpenMM State from a serialised XML file
-- no default, error if undefined
state = { xml = "./mol/ala2vac/State.xml" }

-- the number of times convergence to the QSD is obtained by performing a F-V cycle
--  each time N (being the number of replicas) initial conditions are generated and saved
-- no default, error if undefined
numSteps = 2

-- this table holds parameters for the Fleming-Viot particle process
simulation =
{
  -- A Fleming-Viot particle process is used for sampling the QSD,
  -- without any a priori defined decorrelation or dephasing time stage.
  -- Convergence is checked with Gelman-Rubin statistics
  -- a frequency (in steps) at which to verify if the system left the current state during FV procedure
  checkFV = 250, -- 0.5 ps
  -- a frequency (in steps) at which to evaluate convergence using the Gelman-Rubin statistics
  checkGR = 10, -- 20 fs
  -- Gelman-Rubin statistics checks convergence of several Observables: we define their name here,
  --  then there should be one function matching this name below in this file
  GRobservables = {"getEpot","getEkin","getPhi","getPsi"},
  -- Gelman-Rubin convergence : 0.01 means 1 %
  GRtol = 0.01
}

----------------------------------------------------------------------------------
-- --------------- IMPLICIT VARIABLES AND FUNCTIONS ------------------------------
----------------------------------------------------------------------------------

-- Together with the previous variables, the following implicit global variables and functions
-- are defined from the c++ code, and are accessible from the code you define below
--
-- ------------------
-- implicit variables (read only) :
-- ------------------
--
-- natoms : number of atoms of the system, read from OMM XML files
--
-- mpi_rank_id   : the id of the MPI process working on this file
-- mpi_num_ranks : the total number of MPI processes running
--
-- epot,ekin,etot : 3 variables containing the values of the potential, kinetic and total energies (kcal/mol),
--  of the system for the current simulation time
--
-- timeStep : the MD timeStep, read from OMM XML files
--
--
-- temperature : T of the system in Kelvin, initial value read from OMM XML files ; use get_temperature() for instantaneous values
--
-- ------------------
-- implicit functions : 
-- ------------------
--
-- exit_from_lua() : call this if it is required to finish the simulation from the lua script : it will terminate MPI properly
--  and flush I/O files ; but it won't perform a DB backup so do it manually before.   
--
-- get_coordinates(n) : function returning a tuple x,y,z containing the coordinates of atom n (in nm)
--  NOTE n is indexed in Lua style starting from 1, internally it will be accessed as (n-1) i.e. C++ style
--
-- get_velocities(n) : function returning a tuple vx,vy,vz containing the velocities of atom n (in nm/picosecond)
--  NOTE n is indexed in Lua style starting from 1, internally it will be accessed as (n-1) i.e. C++ style
--
-- get_all_coordinates() : function returning a table crds containing a safe read-only copy of the coordinates
--  access by using : crds.x[i] crds.y[i] crds.z[i] for respectively getting x,y,z coordinates of atom i 
--  NOTE lua indexing, starting from 1
--
-- get_all_velocities() : same as gel_all_coordinates but for velocities ; returns a table vels such that the access is : vels.x[i], etc.
--
-- get_all_crdvels() : returns a 2-tuple crds,vels containing coordinates and vels : internally it call the 2 above defined functions
--
-- get_pbc() : returns periodic boundary conditions as a table pbc with members pbc.a pbc.b pbc.c, each being a xyz vector (i.e. pbc.a.x pbc.a.y pbc.a.z) 
--
-- set_pbc(pbc) : set the openmm periodic boundary conditions to be used, the atgument is a table pbc a described above
--
-- NOTE possibly slow (especially if a copy to a OCL or CUDA device is required), use rarely for performance
-- set_all_coordinates(crds)  : uses a crds tables (as described above) and use it for setting c++/openMM coordinates
-- set_all_velocities(vels)   : the same with velocities
-- set_all_crdvels(crds,vels) : does with one function only what the 2 above defined do
--
-- get_mass(n) : function returning the mass of atom n in a.m.u.
--  NOTE n is indexed in Lua style starting from 1, internally it will be accessed as (n-1) i.e. C++ style
--
-- get_temperature() : get instantaneous value of the temperature in K
--
-- get_COM() : function returning the center of mass of the system as a tuple x,y,z
--
-- get_COM_idxs(idxs) : function returning the center of mass of a subset of the system as a tuple x,y,z
--  NOTE this time idxs is indexed directly in C++ style
--  for example get_COM_idxs({1,2,3}) to get COM of atoms 1, 2 and 3  (C++ : atoms 0, 1, 2)
--
-- get_minimised_energy(tolerance,maxSteps) : this function returns the minimised energy of the system, using the OpenMM L-BFGS minimiser
--  note that coordinates are not affected, it just returns the minimum epot of the bassin in which dynamics currently evolves
--  it returns a 3-tuple ep,ek,et (potential, kinetic and total energy)
--
-- get_minimised_crdvels(tolerance,maxSteps) : this function returns a 2-tuple (crds,vels) containing
--  a copy of coordinates and velocities after minimisation.
--  crds and vels are both tables with x,y,z members, each of size natoms,  : e.g. crds.x[i] returns the x coordinate of atom i, idem for vels.x[i]
--  note that C++/OpenMM coordinates are not modified if modifying this table : this is a safe read-only copy
--  NOTE lua indexing, starting from 1
--
-- hr_timer() : returns a variable representing a c++ high precision timer : can be used for measuring execution time.
--  do not try to modify it or even read it, it should only be used as argument for the following hr_timediff_* functions.
--
-- hr_timediff_ns(before,after) : returns the time difference in nanoseconds between two hr_timer() 'before' and 'after' : usage:
--
--      local bf = hr_timer()
--      function_to_profile()
--      local af = hr_timer()
--      print('Exec. time of function function_to_profile() is (ns) : ',hr_timediff_ns(bf,af))
--
-- hr_timediff_us() and hr_timediff_ms() : same as above but exec time is returned respectively in microseconds and milliseconds
--
-- sleep_for_ms(t) or sleep_for_us(t) or sleep_for_ns(t) : will force the current replica to sleep (do nothing)
--                                                         for a time t (unsigned 64 bit integer), respectively in micro, milli or nano seconds

--------------------------------------------------------------------------------------
-- --------------- USER DEFINED VARIABLES AND FUNCTIONS ------------------------------
--------------------------------------------------------------------------------------

-- Some of the following VARIABLES and FUNCTIONS are mandatory and called from C++ (if it is the case it is explicitly documented)
-- If not they can be restricted to this file using the local keyword

-- Define here local variables and functions used later within state_init() and check_state_left()

---------------------------------------------------------------------------------------
-- --------------- FUNCTIONS DEFINING A METASTABLE STATE ------------------------------
---------------------------------------------------------------------------------------

-- TWO functions, state_init() and check_state_left(), will be called from c++ code to know if the 
--  dynamics left the current state. You are free to define the state in any way, using variables defined explicitly in this file
--  or implicitly (c++ interface, see above).

-- Define here local variables and functions used later within state_init() and check_state_left()

-- atom index definition of each angle (starting at 1, lua style)
local phi_def = {5,7,9,15}
local psi_def = {7,9,15,17}

local fromState,toState = 'unknown','unknown'

-- calculates a dihedral angle between 2 planes
--  idx contains indices of 4 atoms used for defining the 2 planes
--  if crds==nil then cordinates retrieved using get_coordinates, otherwise they are read from this table crds
local function calcDihe(idx,crds)
  
  -- multiplies a vector by a scalar : c[.] = vec[.] * scalar
  local function mulScalVec(scalar,vec)
    local c={0.0,0.0,0.0}
    c[1] = scalar*vec[1]
    c[2] = scalar*vec[2]
    c[3] = scalar*vec[3]
    return c
  end
  
  -- returns dot product ; expects 2 vectors of length 3
  local function dotProduct(a,b)
    return (a[1]*b[1] + a[2]*b[2] + a[3]*b[3])
  end
  
  -- returns the vector corresponding to the cross product of u and v ; length 3
  local function crossProduct(a,b)
    local c={0.0,0.0,0.0}
    c[1] = a[2]*b[3] - a[3]*b[2]
    c[2] = a[3]*b[1] - a[1]*b[3]
    c[3] = a[1]*b[2] - a[2]*b[1]
    return c
  end
  
  -- returns norm of vector
  local function vecNorm(v)
    return math.sqrt(dotProduct(v,v))
  end
  
  -- see wikipedia : https://en.wikipedia.org/wiki/Dihedral_angle#Calculation_of_a_dihedral_angle
  -- Any plane can  be described by two non-collinear vectors lying in that plane;
  -- taking their cross product yields a normal vector to the plane.
  -- Thus, a dihedral angle can be defined by three vectors, b1, b2 and b3,
  -- forming two pairs of non-collinear vectors.
  
  local x1,y1,z1 = 0.,0.,0.
  local x2,y2,z2 = 0.,0.,0.
  if(crds==nil) then
    x1,y1,z1 = get_coordinates(idx[1])
    x2,y2,z2 = get_coordinates(idx[2])
  else
    x1,y1,z1 = crds.x[idx[1]],crds.y[idx[1]],crds.z[idx[1]]
    x2,y2,z2 = crds.x[idx[2]],crds.y[idx[2]],crds.z[idx[2]]
  end
  
  local b1 = {x2-x1,y2-y1,z2-z1}
  b1 = mulScalVec(-1.0,b1)
  
  if(crds==nil) then
    x1,y1,z1 = get_coordinates(idx[2])
    x2,y2,z2 = get_coordinates(idx[3])
  else
    x1,y1,z1 = crds.x[idx[2]],crds.y[idx[2]],crds.z[idx[2]]
    x2,y2,z2 = crds.x[idx[3]],crds.y[idx[3]],crds.z[idx[3]]
  end
  
  local b2 = {x2-x1,y2-y1,z2-z1}
  
  if(crds==nil) then
    x1,y1,z1 = get_coordinates(idx[3])
    x2,y2,z2 = get_coordinates(idx[4])
  else
    x1,y1,z1 = crds.x[idx[3]],crds.y[idx[3]],crds.z[idx[3]]
    x2,y2,z2 = crds.x[idx[4]],crds.y[idx[4]],crds.z[idx[4]]
  end

  local b3 = {x2-x1,y2-y1,z2-z1}
  
  -- cross-product between b1 and b2
  local cp12 = crossProduct(b1,b2)
  
  -- and between b3 and b2
  local cp32 = crossProduct(b3,b2)
  
  -- cp between the 2 normal vectors
  local cpcp = crossProduct(cp12,cp32)
  
  local y = dotProduct(cpcp, b2)*(1.0/vecNorm(b2))
  local x = dotProduct(cp12, cp32)
  local dihe = math.atan2(y,x) * 180.0/math.pi
  
  return dihe
  
end

-- this function is mandatory and called from C++, program will fail if not defined
--  it should take no arguments
--  it returns nothing
-- Use it if you have global variables used in check_state_left() (or other functions) that you need to initialise
function state_init()

  -- value of the phi and psi dihedral angles at initialisation
  local phi = calcDihe(phi_def,nil)
  local psi = calcDihe(psi_def,nil)
  
  local domain = nil
  if( (phi > 0.0 and phi < 120.0) and (psi < 0.0 and psi > -150.0) ) then
    domain = 'C_ax'
  else
    domain = 'C_eq'
  end

  fromState = domain
  toState   = domain
  
  print("Initial state is: "..domain.." {"..phi.." "..psi.."} ")
  
end

-- You may create as many functions as you want and call them from check_state_left(),
--  but the c++ code will in the end only call check_state_left()

-- this function is mandatory and called from C++, program will fail if not defined
--  it should take no arguments
--  it should return a boolean : true in case the dynamics left the state, false otherwise
function check_state_left()

  -- value of the phi and psi dihedral angles of ala2 at a given time value
  local phi = calcDihe(phi_def,nil)
  local psi = calcDihe(psi_def,nil)

  local escaped = false
  
  local domain = nil
  -- first check if within a rectangular domain around C_ax
  if( (phi > 0.0 and phi < 120.0) and (psi < 0.0 and psi > -150.0) ) then
    domain = 'C_ax'
  else
    domain = 'C_eq'
  end
  
  toState = domain
  if(fromState == toState) then escaped = false else escaped = true end
  
  return escaped

end

-------------------------------------------------------------------
-- --------------- STORAGE FUNCTIONS ------------------------------
-------------------------------------------------------------------

-- The SQLite3 interface code was compiled in the main programm (libsqlite3 required on the system when compiling)
-- Therefore we can drive a database from this lua script !

local insert_statement_states  = [[ INSERT INTO CONDITIONS_LIST(COND_ID,NREPS,STATE,TAU_FV)
                                    VALUES ($condid,$nreps,$state,$tau); ]]

local insert_statement_crdvels = [[ INSERT INTO CRD_VELS (COND_ID,REP_ID,ATOM_ID,X,Y,Z,VX,VY,VZ)
                                    VALUES ($condid,$repid,$atid,$x,$y,$z,$vx,$vy,$vz); ]]

-- If database is locked by another replica already attempting a write operation, this makes sure we try again indefinitely but after waiting a bit
local function busy_handler(udata,n_retries)
--   sleep_for_us(1)
  return 1
end

local db = nil

-- database initially created and tables created by rank 0 only, the others just open it when required
if(mpi_rank_id == 0) then
  
  print("Rank 0 creating the database ...")
  
  db = sqlite3.open("FV.db")

  db:exec("PRAGMA journal_mode=WAL;")
  db:exec("BEGIN TRANSACTION;")
  
  -- create the states table : it contains the escape time for this state
  db:exec[[ CREATE TABLE CONDITIONS_LIST(
            COND_ID   INTEGER NOT NULL PRIMARY KEY,
            NREPS     INTEGER,
            STATE     TEXT,
            TAU_FV    REAL); ]]

  -- create the crdvels table : it contains coordinates and velocities of the system for a given 'states' record
  db:exec[[ CREATE TABLE CRD_VELS(
            CV_ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            COND_ID INTEGER,
            REP_ID  INTEGER,
            ATOM_ID INTEGER,
            X   REAL,
            Y   REAL,
            Z   REAL,
            VX  REAL,
            VY  REAL,
            VZ  REAL,
            UNIQUE (COND_ID, REP_ID, ATOM_ID)); ]]
    
  db:exec("END TRANSACTION;")
  
  db:close()
  
end

local cond_id = 0
local rep_id = mpi_rank_id
local nreps = mpi_num_ranks

function save_FV_initial_conditions(tauTime)

  local start_t = hr_timer()
  
  print("Rank ",rep_id," accessing the database ...")

  db = sqlite3.open("FV.db")
  
  db:busy_handler(busy_handler)

  db:exec("PRAGMA journal_mode=WAL;")
  
  local stmt = nil
  
  if(mpi_rank_id == 0)
  then
    db:exec("BEGIN TRANSACTION;")
    stmt = db:prepare(insert_statement_states)
    
    stmt:bind_names{condid=cond_id,nreps=nreps,state=fromState,tau=tauTime}
    stmt:step()
    
--     if(ret ~= sqlite3.DONE)
--     then
--       print("SQLITE3 error in insert_statement_states ; code = ",ret," -> ",db:errmsg())
--       --       exit_from_lua()
--     end

    stmt:finalize()
    db:exec("END TRANSACTION;")
  end
  
  -- to avoid quite poor I/O perfs we ask each replica to sleep for a short time
  --  in order to not have all of them trying to modify the database at the same time
  sleep_for_ms(rep_id+1)
  
  db:exec("BEGIN TRANSACTION;")
  stmt = db:prepare(insert_statement_crdvels)
  
  for n=1,natoms
  do
    local x,y,z = get_coordinates(n)
    local vx,vy,vz = get_velocities(n)

    stmt:bind_names{ condid=cond_id, repid=rep_id, atid=n, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz }
    stmt:step()
    
--     if(ret ~= sqlite3.DONE)
--     then
--       print("SQLITE3 error in insert_statement_crdvels ; code = ",ret," -> ",db:errmsg())
--       --       exit_from_lua()
--     end

    stmt:reset()

  end
  
  stmt:finalize()
  
  db:exec("END TRANSACTION;")
  
  db:close()
  
  cond_id = cond_id + 1
  
  local end_t = hr_timer()
  
  print("Execution of save_FV_initial_conditions(...) on replica ",rep_id," took ",
        hr_timediff_ms(start_t,end_t)," ms")
  
end

--------------------------------------------------------------------------------------------------------------
-- --------------- GELMAN RUBIN FUNCTIONS ESTIMATING OBSERVABLES ------------------------------
--------------------------------------------------------------------------------------------------------------

-- Define a function for calculating the value of each Observable
-- Those functions should:
-- 1) take no arguments 
-- 2) return a double precision (any Lua numeric value is returned as a double precision)
--
-- Those GR functions are called from C++ if they were listed in simulation.GRobservables
--

-- Definition of the "getEpot" and "getEkin" observables used for Gelman-Rubin statistics
-- Just returns value of the potential and kinetic energy
function getEpot()
  return epot
end

function getEkin()
  return ekin
end

function getEtot()
  return etot
end

-- Definition of the "getPhi" and "getPsi" Observable used for Gelman-Rubin statistics
-- Returns current value of the phi and psi dihedral angles
function getPhi()
  return calcDihe(phi_def,nil)
end

function getPsi()
  return calcDihe(psi_def,nil)
end

--------------------------------------------------------------------------------------------------------------
    

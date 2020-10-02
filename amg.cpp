#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <functional>
#include <tuple>
#include <string>
#include <fstream>
#include <algorithm>
#include <utility>
#include <assert.h>
#include <limits>
#include <math.h>

using namespace std;

#define NR 100
#define NZ 200


enum Status{
  UNDECIDED=0,
  COARSE,
  SFINE, //strongly connected to coarse point
  WFINE, //not strongly connected to coarse poit, but ensure strong connection to fine point
  BOUNDARY
};

enum ConnType{
  WEAK=0,
  STRONG,
};

//TODO introduce base line class for matrices that do not need coarsening
//then create the following line class by inheritance
//saves some memory
struct line{
  int ind;
  double val;
  Status status;

  vector<tuple<int,double,ConnType>> conns;

  line():ind(-1),val(0),status(UNDECIDED){}

  line(int _ind, double _val, Status stat, std::vector<std::tuple<int,double,ConnType>> _conns = std::vector<std::tuple<int,double,ConnType>>())
    : ind(_ind),val(_val),status(stat){
    conns = _conns;
  }
};

typedef std::vector<line> Matrix;

//matrix vector multiplication for square matrices
//XXX runtime critical
std::vector<double> matrix_vector(const Matrix& matrix, const std::vector<double>& vec){
  std::vector<double> result(matrix.size(),0);

  for(auto& ln : matrix){
    //diagonal element
    result[ln.ind] += ln.val * vec[ln.ind];
    for(auto& cn:ln.conns){
      int i = std::get<int>(cn);
      double val = std::get<double>(cn);
      result[ln.ind]+=val*vec[i];
    }
  }
  return result;
}

//calculate max norm of the residual
std::pair<int,double> calc_max_norm_res(const Matrix& matrix, const std::vector<double>& rhs, const std::vector<double>& phi){
  double error=0;
  int k=0;
  std::vector<double> res = matrix_vector(matrix,phi);
  for(auto i=0u;i<res.size();++i){
    double new_error=fabs(res[i] - rhs[i]);
    if( new_error > error){ 
      error=new_error;
      k=i;
    }
  }
  return std::make_pair(k,error);
}



void set_matrix_connections(Matrix& matrix, double thresh){
  //check if connections are strong or weak

  std::vector<double> max_conns(matrix.size(),0);
  for(auto i=0u;i<matrix.size();++i){
    line& ln=matrix[i];
    for(auto& cn:ln.conns){
      double val=std::get<double>(cn);
      if(val<max_conns[i]) max_conns[i]=val;
    }
  }

  //get strong transpose connections
  for(auto& ln : matrix){
    for(auto& cn : ln.conns){
      int ind=std::get<int>(cn);
      line& c_ln=matrix[ind];
      for(auto& conn : c_ln.conns){
	if(std::get<int>(conn) != ln.ind) continue;
	double val=std::get<double>(conn);
	if(val <= thresh*max_conns[ind]) std::get<ConnType>(cn) = STRONG;
      }
    }
  }
}

void matrix_setup(Matrix& matrix, int Nr, int Nz, double thresh){
  int Ng=Nr*Nz;
  if(!matrix.empty()) matrix.clear();

  matrix.resize(Ng);

  for(auto r=0;r<Nr;++r){
    for(auto z=0;z<Nz;++z){
      int ind=r*NZ+z;
      line& ln = matrix[ind];
      ln.ind=ind;
      ln.status=UNDECIDED;

      if(z==0 || r==0 || r==Nr-1 || z==Nz-1){ 
	ln.val=1;
	ln.status=BOUNDARY;
      }
      else{
	ln.val=4;
	ln.conns.emplace_back(ind-Nz,-1,WEAK);
	ln.conns.emplace_back(ind-1,-1,WEAK);
	ln.conns.emplace_back(ind+1,-1,WEAK);
	ln.conns.emplace_back(ind+Nz,-1,WEAK);
      }
    }
  }
  set_matrix_connections(matrix,thresh);
}


void fill_rhs(vector<double>& dens, int Nr, int Nz){
  if(!dens.empty()) dens.clear();

  int Ng = Nr*Nz;
  dens.resize(Ng);

  for(auto r=0;r<Nr;++r){
    for(auto z=0;z<Nz;++z){
      int i=r*Nz+z;
      //dens[i] = 10 * (r+z)/(double) Ng;
      if(r==0 || z==0 || r==Nr-1 || z==Nz-1) dens[i]=0;
      else dens[i]=1;
    }
  }
}


void print_coarsening(const Matrix& matrix, int Nr, int Nz){
  cout<<endl<<endl;

  for(auto r=0;r<Nr;++r){
    for(auto z=0;z<Nz;++z){
      int i=r*Nz+z;
      cout<<setw(6)<<(int) matrix[i].status;
    }
    cout<<endl;
  }
  cout<<endl<<endl;
}


void print_matrix(const Matrix& matrix){
  if(!matrix.empty()){
    cout<<endl;
    for(auto i=0u;i<matrix.size();++i){
      auto& ln = matrix[i];
      std::cout<< ln.ind<<"   "<<std::setw(6)<< ln.val<<std::setw(6)<< (int) ln.status<<std::endl;
      for(auto& cn : ln.conns){
	std::cout<<"-->"<<"    "<<std::get<int>(cn)<<"    "<<std::setw(6)<<std::get<double>(cn)<<std::endl;
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }else{
    std::cout<<"matrix to print is empty"<<std::endl;
  }
}


void print_vector(const std::vector<double>& vec, int Nr, int Nz){
  cout<<endl<<endl;

  for(auto r=0;r<Nr;++r){
    for(auto z=0;z<Nz;++z){
      int i=r*Nz+z;
      cout<<setw(10)<<vec[i];
    }
    cout<<endl;
  }
  cout<<endl<<endl;
}

//these iterations should only be used for smoothing as no error check is used
//sor iteration with system matrix "matrix", rhs "rhs"
//and guess phi which should be 0 for coarse levels which afterwards stores the solution
//XXX runtime critical
void sor(const Matrix& matrix, const std::vector<double>& rhs, std::vector<double>& phi, int maxstep=5, double omega=1){
  assert(matrix.size() == rhs.size() && "matrix & vector size dont match in sor(...)");
  assert(matrix.size() == phi.size() && "matrix & vector size dont match in sor(...)");
  int steps=0;

  auto doProcessing = [&](int i){ 
    const line& ln = matrix[i];
    //apply constant bc if necessary
    if (ln.val == 1){ 
      phi[i] = rhs[i];
    }else{
      double phi0 = rhs[i];
      //matrix vector multiplication
      phi0 -= ln.val*phi[ln.ind];
      for(auto & cn : ln.conns) phi0 -= std::get<double>(cn) * phi[std::get<int>(cn)];
      phi0/=ln.val;
      phi[i] += omega * phi0;
    }
  };

  do{
#if 1
    //checkerboard pattern
    for(auto i=0u;i<rhs.size();i+=2) doProcessing(i);
    for(auto i=1u;i<rhs.size();i+=2) doProcessing(i);
#else
    //symmetric pattern
    for(auto i=0u;i<rhs.size();++i) doProcessing(i);
    for(auto i=rhs.size()-1;i>=0;--i) doProcessing(i);
#endif
    ++steps;
  }while(steps < maxstep);
  //cout<<"sor: error="<<error<<"  steps="<<steps<<endl;
}


//jacobi iteration with system matrix "matrix", rhs "rhs", damping parameter omega
//and guess phi which should be 0 for coarse levels which afterwards stores the solution
//XXX runtime critical
void jacobi(const Matrix& matrix, const std::vector<double>& rhs, std::vector<double>& phi, int maxstep=5, double omega=0.6666){
  assert(matrix.size() == rhs.size() && "matrix & vector size dont match in jacobi(...)");
  assert(matrix.size() == phi.size() && "matrix & vector size dont match in jacobi(...)");
  int steps=0;
  std::vector<double> phi_old(phi);

  auto doProcessing = [&](int i){ 
    const line& ln = matrix[i];
    //apply constant bc if necessary
    if (ln.val == 1){ 
      phi[i] = rhs[i];
    }else{
      double phi0 = rhs[i];
      //matrix vector multiplication
      phi0 -= ln.val*phi[ln.ind];
      for(auto & cn : ln.conns) phi0 -= std::get<double>(cn) * phi_old[std::get<int>(cn)];
      phi0/=ln.val;
      phi[i] += omega * phi0;
    }
  };

  do{
    for(auto i=0u;i<rhs.size();++i) doProcessing(i);
    std::copy(phi.begin(),phi.end(),phi_old.begin());
    ++steps;
  }while(steps < maxstep);
  //cout<<"jacobi: error="<<error<<"  steps="<<steps<<endl;
}


//non pre-conditioned cg method to solve the AMG system on the lowest level
//(should be) guaranteed to converge after dim(phi) steps
void cg(const Matrix& matrix, const std::vector<double>& rhs, std::vector<double>& phi, int maxsteps, double acc=1e-12){
  assert(matrix.size() == rhs.size() && "matrix & vector size dont match in cg(...)");
  assert(matrix.size() == phi.size() && "matrix & vector size dont match in cg(...)");

  double error=0;
  int steps=0;

  auto scalar_product = [](const std::vector<double>& v1, const std::vector<double>& v2){
    assert(v1.size() == v2.size() && "vector dimensions dont match in scalar_product");
    double result=0;
    for(auto i=0u;i<v1.size();++i) result+=v1[i]*v2[i];
    return result;
  };

  std::vector<double> residual=matrix_vector(matrix,phi);
  for(auto i=0u;i<residual.size();++i) residual[i] = rhs[i] - residual[i];

  std::vector<double> direction=residual;
  std::vector<double> mvprod(direction.size(),0);
  std::vector<double> new_residual(residual);

  for(auto i=0;i<maxsteps;++i){
    error=0;
    ++steps;

    //save this mvproduct
    mvprod = matrix_vector(matrix,direction);

    //find phi in search direction and update gradient and residual
    double alpha = scalar_product(residual,residual);
    alpha /= scalar_product(direction,mvprod);
    for(auto k=0u;k<phi.size();++k) phi[k]+=alpha*direction[k];
    for(auto k=0u;k<residual.size();++k){ 
      new_residual[k] = residual[k] - alpha*mvprod[k];
      //find maximum norm of residual
      if(fabs(new_residual[k])>error) error=fabs(new_residual[k]);
    }
    if(error<acc) break;

    //update search direction
    double beta = scalar_product(new_residual,new_residual);
    beta /= scalar_product(residual,residual);
    for(auto k=0u;k<direction.size();++k) direction[k] = new_residual[k] + beta*direction[k];
    std::copy(new_residual.begin(),new_residual.end(),residual.begin());
  }

  //cout<<"cg: error="<<error<<"  steps="<<steps<<endl;
}


//TODO move to new file/header when done
//TODO find more elegant way of constructing hierarchy other than copying (not rly critical for runtime)
struct AMGlvl{
  Matrix matrix;
  Matrix interpolation;
  int level;
  int max_level;
  std::vector<double> phi;
  std::vector<double> rhs;

  //add Matrix&& constructor so that initially calculated matrix doesnt have to be deleted
  AMGlvl(Matrix& _matrix, int lvl):level(lvl){
    matrix = _matrix;
    interpolation = Matrix();
    phi.resize(matrix.size(),0);
    rhs.resize(matrix.size(),0);
  }

  AMGlvl(Matrix& _matrix, Matrix& _interpolation, int lvl):level(lvl){
    matrix = _matrix;
    interpolation = _interpolation;
    phi.resize(matrix.size(),0);
    rhs.resize(matrix.size(),0);
  }

  AMGlvl():level(-1),max_level(-1){
  }

  int size(){
    return matrix.size();
  }

  //for each point the sum of inteprolation weights should be 1
  double interpolation_check(){
    if(!interpolation.empty()){
      double error=0;
      double sum=0;

      for(auto i=0u;i<interpolation.size();++i){
	auto& ln = interpolation[i];
	sum=0;
	for(auto& cn:ln.conns) sum+=std::get<double>(cn);
	if(sum!=0) sum=1-sum;
	std::cout<<"i="<<i<<" sum="<<sum<<std::endl;
	if(sum>error) error=sum;
      }

      return sum;
    }else return 0;
  }
};


typedef std::vector<AMGlvl> AMGhierarchy;


void coarsen_matrix(Matrix& matrix, double beta){
  typedef std::set<std::tuple<int,double>,bool(*)(const std::tuple<int,double>& t1, const std::tuple<int,double>& t2)> SetWeights;
  typedef SetWeights::iterator iterator_tp;

  //set combining weight and index of each grid point which is used for selection of coarse points
  //use std::set which is ordered with passed ordering function
  SetWeights weights([](const std::tuple<int,double>& t1, const std::tuple<int,double>& t2){
      double val1 = std::get<double>(t1);
      double val2 = std::get<double>(t2);
      int ind1 = std::get<int>(t1);
      int ind2 = std::get<int>(t2);
      //otherwise sort by descending weight primarily and ascending index secondarily
      if(val1!=val2) return val1 > val2;
      else return ind1 < ind2;
    }
  );


  //array associating iterator and array_index of each point for fast element access
  std::vector<iterator_tp> weights_it(matrix.size());

  //necessary functions for coarsening algorithm
  auto calc_weight = [&](int i){
    double wt = 0;
    auto& pt = matrix[i];

    if(pt.status==UNDECIDED){
      for(auto& cn : pt.conns){
	int ind = std::get<int>(cn);
	if(matrix[ind].status==SFINE && std::get<ConnType>(cn)==STRONG) wt += 1;
	else if(matrix[ind].status==UNDECIDED && std::get<ConnType>(cn)==STRONG) wt += 1;
      }
    }
    return wt;
  };

  auto find_matrix_val = [&matrix](int i, int j){
    if(i==j) return matrix[i].val;
    else{
      for(auto& cn:matrix[i].conns){
	if(std::get<int>(cn)==j) return std::get<double>(cn);
      }
      return (double) 0;
    }
  };

  //returns coupling strength of j to Coarse_points in C(i) and from i to j
  auto calc_coupling_strength = [&] (int j, int i){
    //coupling fro mj to C(i)
    double strength_j_to_ci = 0;

    //not sure if diagonal values for i and j should be considered
    double max_j_entry = fabs(matrix[j].val);
    double max_i_entry = fabs(matrix[i].val);

    for(auto& cn : matrix[j].conns){
      double val = std::get<double>(cn);
      if(fabs(val) > max_j_entry) max_j_entry = fabs(val);
    }

    //add entries in coarse neighborhood of 
    for(auto& cn : matrix[i].conns){
      int ind = std::get<int>(cn);
      double val = std::get<double>(cn);
      //find max i entry
      if(fabs(val) > max_i_entry) max_i_entry = fabs(val);

      //calc coupling strength
      if(std::get<ConnType>(cn) == STRONG && matrix[ind].status==COARSE){
	strength_j_to_ci -= find_matrix_val(j,ind);
      }
    }
    assert(max_j_entry != 0 && max_i_entry != 0 && "max entry is 0 in calc_coupling_strength");
    strength_j_to_ci /= max_j_entry;

    //now calc strength of coupling from i to j
    double strength_i_to_j = find_matrix_val(i,j)/max_i_entry;

    return std::make_pair(strength_j_to_ci,strength_i_to_j);
  };


  //remove points in list to_update
  auto erase_weights = [&](std::vector<int> to_erase){
    for(auto ind : to_erase){ 
      iterator_tp& it = weights_it[ind];
      weights.erase(it);
      it = weights.end();
    }
  };

  //update points in list to_update
  auto update_weights = [&](std::vector<int> to_update){
    //erase old tuple, update weight of index an insert new tuple into weights, then update iterator
    for(auto ind : to_update){
      iterator_tp& it = weights_it[ind];
      double wt = calc_weight(ind);

      weights.erase(it);
      it = weights.insert(std::make_tuple(ind,wt)).first;
    }
  };

  auto add_coarse_point = [&](int i){
    //std::cout<<"adding point "<<std::get<int>(*weights.begin())<<"  with weight "<<std::get<double>(*weights.begin())<<"  remaining before erasing current "<<weights.size()<<std::endl;
    std::vector<int> to_update;
    std::vector<int> to_erase;
    auto& coarse_point = matrix[i];

    //for second pass coarsening
    if(coarse_point.status != UNDECIDED){
      coarse_point.status=COARSE;
    //first pass coarsening
    }else{
      coarse_point.status=COARSE;
      to_erase.push_back(i);

      //first mark all strongly connected points as SFINE, then add them to erase them from vector
      for(auto& coarse_conn : coarse_point.conns){
	int ind = std::get<int>(coarse_conn);
	auto& fine_point = matrix[ind];
	if(fine_point.status == UNDECIDED && std::get<ConnType>(coarse_conn) == STRONG){
	  fine_point.status = SFINE;
	  if(weights_it[ind] != weights.end()) to_erase.push_back(ind);
	}
      }

      //erase those points from std::multiset
      erase_weights(to_erase);

      //update the weights of weakly connected neighbors
      //i's conncetions as to_update if they are undecided. 
      for(auto& coarse_conn : coarse_point.conns){
	int f_ind = std::get<int>(coarse_conn);
	auto& fine_point = matrix[f_ind];
	if(weights_it[f_ind] != weights.end()) to_update.push_back(f_ind);

	for(auto& fine_conn : fine_point.conns){
	  int ind = std::get<int>(fine_conn);
	  if(matrix[ind].status == UNDECIDED){ 
	    if(weights_it[ind] != weights.end()){
	      if(std::find(to_update.begin(),to_update.end(),ind) == to_update.end()) to_update.push_back(ind);
	    }
	  }
	}
      }
#if 0
      for(auto& ind : to_update)
	std::cout<<"   updating point "<<ind<<"  old weight "<<std::get<double>(*weights_it[ind])<<"  new_weight "<<calc_weight(ind)<<std::endl;
      for(auto& ind : to_erase)
	std::cout<<"   erasing  point "<<ind<<std::endl;
#endif
      update_weights(to_update);
    }
  };


  //actual coarsening algorithm
  //calculate weights and insert them into the set which takes care of the order
  for(auto i=0u;i<weights_it.size();++i){
    double wt = calc_weight(i);
    std::pair<iterator_tp,bool> it_pair = weights.insert(std::make_tuple(i,wt));
    weights_it[i]=it_pair.first;
    if(it_pair.second == false){ 
      std::cout<<"point "<<i<<" with weight "<<wt<<" not inserted into set!"<<std::endl;
      weights_it[i]=weights.end();
    }
  }

  //erase 0 weight entries and mark them as boundary entries
  for(auto& it:weights_it){
    if(std::get<double>(*it)==0){
      matrix[std::get<int>(*it)].status = BOUNDARY;
      weights.erase(it);
      it = weights.end();
    }
  }

  //first pass of coarsening
  //creates maximal set in the sense that no coarse point strongly depends on other coarse point
  do{
    add_coarse_point(std::get<int>(*weights.begin()));
  //}while(!weights.empty());
  }while(std::get<double>(*weights.begin()) != 0);



  //second pass of coarsening ensures that strong F-F connections require a common coarse point
  for(auto i=0u;i<matrix.size();++i){
    auto& ln = matrix[i];
    if(ln.status==UNDECIDED){ 
      std::cout<<"still undecided in second pass of coarsening. This should not happen!"<<std::endl;
      exit(-1);
    }
    else if(ln.status==COARSE) continue;
    //only add additional coarse points near fine points
    else{
      int coarse_point_added=0;

      for(auto& cn : ln.conns){
	int ind = std::get<int>(cn);
	if(matrix[ind].status==COARSE || std::get<ConnType>(cn)!=STRONG) continue;

	auto coupling_strength = calc_coupling_strength(ind,i);
	if(coupling_strength.first <= beta* coupling_strength.second){
	  if(coarse_point_added == 0){
	    matrix[ind].status=COARSE;
	    coarse_point_added=ind;
	  }else{
	    matrix[i].status=COARSE;
	  }
	}
      }
    }
  }

  //for debugging. no point should be UNDECIDED at this point
#if 0
  for(auto& ln : matrix){
    if(ln.status==UNDECIDED){
      std::cout<<"Index "<<ln.ind<<" was not coarsened despite strong connections to other points. This should not happen in AMG!"<<std::endl;
      exit(-1);
    }
  }
#endif
}


//creates interpolation from coarser level to the level of matrix
//using direct interpolation as defined by K. Stueben
//direct interpolation is good for zero row sum matrices where each fine point needs at least 1 strong coarse neighbor connection
// means no point is (WFINE)
Matrix create_interpolation_matrix(const Matrix& matrix){
  Matrix interpolation;

  //map of coarse points between old and new grid
  std::vector<int> coarse_map(matrix.size(),-2);
  int coarse_ctr=0;
  for(auto i=0u;i<matrix.size();++i){
    if(matrix[i].status == COARSE){
      coarse_map[i]=coarse_ctr++;
    }
  }

  //calc weight for one interpolation matrix element. i is row index, j is column index (if i!=j -> goes into conns)
  //direct interpolation
#if 0
  //uncomment to remove compile warning
  auto calc_interpolation_weights = [&](int i){
    std::vector<std::tuple<int,double,ConnType>> weights;
    const line& ln = matrix[i];
    double sum_pos_interpolation_points = 0;
    double sum_neg_interpolation_points = 0;
    double alpha = 0;
    double beta = 0;

    for(auto& cn : ln.conns){
      double val = std::get<double>(cn);
      int ind = std::get<int>(cn);

      if(val>0){
	if(matrix[ind].status == COARSE && std::get<ConnType>(cn)==STRONG) sum_pos_interpolation_points += val;
	beta += val;
      }else if(val<0){
	if(matrix[ind].status == COARSE && std::get<ConnType>(cn)==STRONG) sum_neg_interpolation_points += val;
	alpha += val;
      }
    }

    if(sum_neg_interpolation_points!=0) alpha /= sum_neg_interpolation_points;
    else alpha=0;
    if(sum_pos_interpolation_points!=0) beta /= sum_pos_interpolation_points;
    else beta=0;

    //assert((alpha != 0 || beta != 0 ) && "no interpolation points to calc_interpolation_weights");

    for(auto k=0u;k<ln.conns.size();++k){
      auto& cn = ln.conns[k];
      int ind = std::get<int>(cn);
      double val = std::get<double>(cn);

      //the conns are given in the coordinates of the coarser matrix
      if(matrix[ind].status == COARSE && std::get<ConnType>(cn)==STRONG){
	if(val<0){
	  double wt = -1 * alpha * val / ln.val;
	  weights.emplace_back(coarse_map[ind],wt,STRONG);
	}else if(val>0){
	  double wt = -1 * beta * val / ln.val;
	  weights.emplace_back(coarse_map[ind],wt,STRONG);
	}
      }
    }
    return line(i,0,SFINE,weights);
  };
#endif


  //TODO move this and in create_coarse into 1 function
  auto find_matrix_val = [&matrix](int i, int j){
    if(i==j) return matrix[i].val;
    else{
      for(auto& cn:matrix[i].conns){
	if(std::get<int>(cn)==j) return std::get<double>(cn);
      }
      return (double) 0;
    }
  };


  //calculate coefficient necessary for improved interpolation, formula
  //c_ij = sum_(k in FINE(i)) ( matrix(i,k)*matrix(k,j) / ((sum_(l in COARSE(i)) (matrix(k,l))) + matrix(k,i))
  
  auto calc_int_coefficient = [&matrix,&find_matrix_val](int i, int j){
    double coeff = 0;
    for(auto& cn : matrix[i].conns){
      int k = std::get<int>(cn);
      if(matrix[k].status==SFINE || matrix[k].status==WFINE){
	double k_l_sum = 0;
	for(auto& cn_coarse : matrix[i].conns){
	  int l = std::get<int>(cn_coarse);
	  if(matrix[l].status==COARSE) k_l_sum += find_matrix_val(k,l);
	}
	coeff += (find_matrix_val(i,k) * find_matrix_val(k,j)) / (k_l_sum + find_matrix_val(k,i));
      }
    }
    return coeff;
  };


  //improved interpolation formula working also for unstructured meshes
  auto calc_interpolation_weights_new = [&](int i){
    std::vector<std::tuple<int,double,ConnType>> weights;
    const line& ln = matrix[i];
    assert((ln.status==SFINE || ln.status==WFINE) && "calcing interpolation weight for non-FINE point");

    //forst calculate coefficient c_ii since it is needed later
    double coeff_ii = calc_int_coefficient(i,i);
    double val_ii   = ln.val;

    for(auto& cn : ln.conns){
      int j = std::get<int>(cn);
      //interpolate from coarse points
      if(matrix[j].status == COARSE){
	double wt = -1*(find_matrix_val(i,j) + calc_int_coefficient(i,j));
	wt /= (val_ii + coeff_ii);
	weights.emplace_back(coarse_map[j],wt,STRONG);
      }
    }
    return line(i,0,SFINE,weights);
  };


  for(auto i=0u;i<matrix.size();++i){
    const line& ln = matrix[i];
    
    if(ln.status == UNDECIDED){
      //this really should not happen
      std::cout<<"trying to interpolate a not properly coarsened matrix!"<<std::endl;
      std::cout<<"ind="<<ln.ind<<"  val="<<ln.val<<std::endl<<"neighbors:"<<std::endl;
      for(auto& cn:ln.conns){
	int ind=std::get<int>(cn);
	double val=std::get<double>(cn);
	std::cout<<"   ind="<<ind<<"  val="<<val<<"  connection_type="<<std::get<ConnType>(cn)<<"  stat="<<matrix[ind].status<<std::endl;
      }
      exit(-1);
    }else if(ln.status == COARSE){
      std::vector<std::tuple<int,double,ConnType>> wt;
      wt.emplace_back(coarse_map[i],1,STRONG);
      interpolation.emplace_back(i,0,COARSE,wt);
    }else if(ln.status == SFINE){
      interpolation.emplace_back(calc_interpolation_weights_new(i));
    }else if(ln.status == WFINE){
      //no point should be WFINE for this type of interpolation
      interpolation.emplace_back(i,0,WFINE);
    }else{
      //should only be boundaries
      interpolation.emplace_back(i,0,BOUNDARY);
    }
  }
  return interpolation;
}


Matrix create_coarse_matrix(const Matrix& matrix, const Matrix& int_matrix, double thresh){
  assert(matrix.size() && matrix.size()==int_matrix.size() && "matrix dimensions dont match to create coarse matrix");
  //map of coarse points between old and new grid
  std::vector<int> coarse_map(matrix.size(),-2);
  int coarse_ctr=0;
  for(auto i=0u;i<matrix.size();++i){
    if(matrix[i].status == COARSE){
      coarse_map[i]=coarse_ctr++;
    }
  }

  auto find_matrix_val = [](const Matrix& matrix, int i, int j){
    if(i==j) return matrix[i].val;
    else{
      for(auto& cn:matrix[i].conns){
	if(std::get<int>(cn)==j) return std::get<double>(cn);
      }
      return (double) 0;
    }
  };


  Matrix coarse_matrix(coarse_ctr);
  for(auto i=0u;i<coarse_matrix.size();++i) coarse_matrix[i].ind=i;

  for(auto k=0u;k<matrix.size();++k){
    const line& k_ln=matrix[k];
    if(k_ln.status==BOUNDARY) continue;

    std::vector<int> k_coarse;
    std::vector<int> l_list(1,k);

    for(auto& cn : k_ln.conns){
      int ind=std::get<int>(cn);
      if(std::get<ConnType>(cn)==STRONG) {
	l_list.push_back(ind);
	if(matrix[ind].status==COARSE){  
	  k_coarse.push_back(ind);
	  //std::cout<<"k="<<k<<" added ind="<<ind<<std::endl;
	}
      }
    }

    for(auto& l : l_list){
      std::vector<int> coarse_list(k_coarse);

      //add l if it is Coarse and not yet added
      if((std::find(coarse_list.begin(),coarse_list.end(),l) == coarse_list.end()) && matrix[l].status==COARSE) 
	coarse_list.push_back(l);

      for(auto& cn:matrix[l].conns){
	int ind = std::get<int>(cn);
	if(std::get<ConnType>(cn)==STRONG && matrix[ind].status==COARSE){ 
	  if(std::find(coarse_list.begin(),coarse_list.end(),ind)==coarse_list.end()){
	    coarse_list.push_back(ind);
	    //std::cout<<"   k="<<k<<" l="<<l<<" added ind="<<ind<<std::endl;
	  }
	}
      }

      for(auto i:coarse_list){
	for(auto j:coarse_list){
	  int i_new=coarse_map[i];
	  int j_new=coarse_map[j];
	  double to_add=0;

	  //calc value to add to new matrix element (i_new,j_new)
	  to_add = find_matrix_val(int_matrix,k,i_new)*find_matrix_val(int_matrix,l,j_new)*find_matrix_val(matrix,k,l);
	  //std::cout<<"added "<<to_add<<" to element via indices i="<<i_new<<" j="<<j_new<<" k="<<k<<" l="<<l<<"  "<<find_matrix_val(int_matrix,k,i_new)<<"  "<<find_matrix_val(int_matrix,l,j_new)<<"  "<<find_matrix_val(matrix,k,l)<<std::endl;

	  if(to_add!=0){
	    //std::cout<<"added "<<to_add<<" to element via indices i="<<i_new<<" j="<<j_new<<" k="<<k<<" l="<<l<<std::endl;
	    line& new_ln = coarse_matrix[i_new];

	    //if diag_element just add to val, otherwise to the right conn
	    if(i_new==j_new) coarse_matrix[i_new].val+=to_add;
	    else{
	      //if cn exists add there, otherwise create
	      bool added=false;
	      for(auto& cn:new_ln.conns){
		if(std::get<int>(cn) == j_new){
		  std::get<double>(cn) += to_add;
		  added=true;
		}
	      }
	      if(!added){
		new_ln.conns.emplace_back(j_new,to_add,WEAK);
	      }
	    }
	  }
	}
      }
    }
  }

  set_matrix_connections(coarse_matrix,thresh);

  return coarse_matrix;
}


//first coarsen matrix of finelvl
//then create interpolation matrix and assign it to the coarser lvl
//then create new system matrix for the new lvl and assign it as well
AMGlvl create_coarser_level(AMGlvl& finelvl, double thresh, double beta){
  coarsen_matrix(finelvl.matrix, beta);
  Matrix new_interpolation = create_interpolation_matrix(finelvl.matrix);
  Matrix coarse_matrix = create_coarse_matrix(finelvl.matrix,new_interpolation,thresh);
  return AMGlvl(coarse_matrix,new_interpolation,finelvl.level+1);
}


AMGhierarchy create_amg_hierarchy(Matrix& _matrix, int max_lvl, int minsize, double thresh, double beta){
  AMGhierarchy hierarchy;

  //initialize finest level
  hierarchy.emplace_back(AMGlvl(_matrix,0));

  for(auto i=1;i<max_lvl && hierarchy.back().size()>minsize; ++i){
    AMGlvl& last_lvl = hierarchy[i-1];
    hierarchy.emplace_back(create_coarser_level(last_lvl,thresh,beta));
  }

  //set maxlevel for each
  std::for_each(hierarchy.begin(),hierarchy.end(),[&hierarchy](AMGlvl& lvl){lvl.max_level=hierarchy.size();});

  return hierarchy;
}


void print_hierarchy(AMGhierarchy& hier, int Nr, int Nz, std::string fn){
  int size = hier.front().size();
  std::vector<int> hmap(size,0);

  for(auto k=1u;k<hier.size();++k){
    Matrix& intp=hier[k].interpolation;
    int ctr=0;

    for(auto i=0;i<size && ctr<static_cast<int>(intp.size()) ;++i){
      if(hmap[i]==static_cast<int>(k)-1){
	line& ln = intp[ctr++];
	if((ln.conns.size() == 1) && std::get<double>(ln.conns.front())==1){
	  hmap[i]=k;
	}
      }
    }
  }

  std::ofstream ofs(fn);
  for(auto r=0;r<Nr;++r){
    for(auto z=0;z<Nz;++z){
      int i=r*Nz+z;
      ofs<<hmap[i]<<" ";
    }
    ofs<<endl;
  }
}

//interpolation from coarser to finer level
//assumes interpolation matrix is passed
//XXX runtime critical
//TODO pass by reference and make function void since all arrays are assumed to be initialized
std::vector<double> interpolation(const Matrix& matrix, const std::vector<double>& vec){
  std::vector<double> int_vec(matrix.size(),0);

  for(auto i=0u;i<matrix.size();++i){
    double& element = int_vec[i];
    for(auto& cn:matrix[i].conns){
      int ind=std::get<int>(cn);
      double val=std::get<double>(cn);
      element += val*vec[ind];
    }
  }
  return int_vec;
}


//restriction from finer to coarser level
//assumes restriction matrix is transpose of interpolation matrix
//XXX runtime critical
//TODO pass by reference and make function void since all arrays are assumed to be initialized
//TODO remove coarse size
std::vector<double> restriction(const AMGlvl& lvl, const std::vector<double>& vec){
  int coarse_size = lvl.matrix.size();
  const Matrix& matrix=lvl.interpolation;
  std::vector<double> res_vec(coarse_size,0);

  for(auto i=0u;i<matrix.size();++i){
    auto& ln = matrix[i];
    for(auto& cn:ln.conns){
      int ind=std::get<int>(cn);
      double val=std::get<double>(cn);
      double& element = res_vec[ind];
      element += vec[i] * val;
    }
  }
  return res_vec;
}


//phi_vec and rhs_vec need to be initialized properly in amg cycle routine
//move "down" from finer to coarser level in the amg hierarchy
double go_down(int i, AMGhierarchy& Hier, double maxerror, int maxsteps){
  AMGlvl& fine_lvl = Hier[i];
  AMGlvl& coarse_lvl = Hier[i+1];
  double error=1;
  if(i==0) error=0;

  assert(i < fine_lvl.max_level-1 && "in AMG::go_down the index was chosen too high");

  //make it better readable
  std::vector<double>& fine_phi = fine_lvl.phi;
  const std::vector<double>& fine_rhs = fine_lvl.rhs;
  std::vector<double>& coarse_rhs = coarse_lvl.rhs;

  //smoothing iteration
  sor(fine_lvl.matrix,fine_rhs,fine_phi,maxsteps);
  
  //calculate residual on current level
  std::vector<double> residual = matrix_vector(fine_lvl.matrix,fine_phi);
  for(auto i=0u;i<residual.size();++i) residual[i] = fine_rhs[i] - residual[i];

  //on finest level calc max residual
  if(i==0){
    for(auto i=0u; i<residual.size();++i){
      double locerror = residual[i]>=0 ? residual[i] : -1*residual[i];
      if(locerror > error) error = locerror;
    }
  }
  
  //restrict residual to next level
  coarse_rhs = restriction(coarse_lvl,residual);

  return error;
}


//move "up" one level in the amg hierarchy
void go_up(int i, AMGhierarchy& Hier, double maxerror, int maxsteps){
  AMGlvl& coarse_lvl = Hier[i];
  AMGlvl& fine_lvl = Hier[i-1];

  assert(i > 0 && "in AMG::go_up the index was chosen to be 0");

  std::vector<double>& coarse_phi = coarse_lvl.phi;
  std::vector<double>& fine_phi = fine_lvl.phi;
  std::vector<double>& fine_rhs = fine_lvl.rhs;
  
  //interpolation to finer lvl
  std::vector<double> phi_correction = interpolation(coarse_lvl.interpolation,coarse_phi);
  
  //add coarse grid correction to current iterate
  for(auto i=0u;i<fine_phi.size();++i) fine_phi[i]+=phi_correction[i];

  //if we are not on finest level, use new first guess during next iteration
  std::fill(coarse_phi.begin(),coarse_phi.end(),0);

  //post smoothing
  sor(fine_lvl.matrix,fine_rhs,fine_phi,maxsteps);
}


//testing purpose only!
//dense calculation of m2^T*m1*m2
#if 0
typedef std::vector<std::vector<double>> densemat;
densemat matrix_vector_to_dense(const Matrix& m1, const Matrix& m2, int m1size1, int m1size2, int m2size1,int m2size2){
  //first convert m1 and m2 to dense matrices then multiply

  auto find_matrix_val = [](const Matrix& matrix, int i, int j){
    if(i==j) return matrix[i].val;
    else{
      for(auto& cn:matrix[i].conns){
	if(std::get<int>(cn)==j) return std::get<double>(cn);
      }
      return (double) 0;
    }
  };

  //initialize matrix
  assert(m1size2 == m2size1 && "cant multiply dense matrices");
  densemat tmp(m1size1);
  std::for_each(tmp.begin(),tmp.end(),[&m2size2](std::vector<double>& v){v.resize(m2size2,0);});

  //calc m1*m2
  for(auto i=0u;i<tmp.size();++i){
    for(auto j=0u;j<tmp[0].size();++j){
      double sum=0;
      for(auto k=0;k<m1size2;++k){
	sum += find_matrix_val(m1,i,k)*find_matrix_val(m2,k,j);
      }
      tmp[i][j]=sum;
    }
  }


  densemat res(m2size2);
  std::for_each(res.begin(),res.end(),[&m2size2](std::vector<double>& v){v.resize(m2size2,0);});
  //calc m2^T*(m1*m2)
  for(auto i=0u;i<res.size();++i){
    for(auto j=0u;j<res[0].size();++j){
      double sum=0;
      for(auto k=0;k<m1size1;++k){
	sum+=0.5 * find_matrix_val(m2,k,i) * tmp[k][j];
      }
      res[i][j]=sum;
    }
  }
  return res;
}


//restrict from level imin up to level imax
void continued_restriction(AMGhierarchy& hier, std::vector<double>& vec, int imin, int imax){
  assert(vec.size() == hier[imin].matrix.size() && "sizes in continued restriction dont match");
  std::vector<std::vector<double>> phi_vec(1,vec);

  for(auto i=imin+1;i<=imax;++i){
    int size = hier[i].matrix.size();
    phi_vec.emplace_back(std::vector<double>(size,0));
  }

  for(auto i=imin;i<imax;++i){ 
    AMGlvl& nextlvl=hier[i+1];
    const std::vector<double>& current_phi = phi_vec[i];
    std::vector<double>& next_phi = phi_vec[i+1];
    next_phi = restriction(nextlvl,current_phi);
#if 1
    for(auto k=0u;k<next_phi.size();++k){
      std::cout<<"k="<<k<<"   phi[k]="<<next_phi[k]<<std::endl;
    }
    std::cout<<endl<<endl;
#endif
  }
  vec = phi_vec.back();
}

//restrict from level imin up to level imax
void continued_interpolation(AMGhierarchy& hier, std::vector<double>& vec, int imax, int imin){
  assert(vec.size() == hier[imax].matrix.size() && "sizes in continued interpolation dont match");

  std::vector<std::vector<double>> phi_vec;
  for(auto i=imin;i<=imax-1;++i){
    int size = hier[i].matrix.size();
    phi_vec.emplace_back(std::vector<double>(size,0));
  }
  phi_vec.push_back(vec);

  for(auto i=imax;i>imin;--i){ 
    AMGlvl& currentlvl=hier[i];
    const std::vector<double>& current_phi = phi_vec[i];
    std::vector<double>& next_phi = phi_vec[i-1];
    next_phi = interpolation(currentlvl.interpolation,current_phi);
#if 1
    for(auto k=0u;k<next_phi.size();++k){
      std::cout<<"k="<<k<<"   phi[k]="<<next_phi[k]<<std::endl;
    }
    std::cout<<endl<<endl;
#endif
  }
  vec = phi_vec.front();
}


void test_iterations(AMGhierarchy& hier, double maxerror, int maxsteps, double omega){
  //testing gauss-seidel iteration
  for(auto& lvl:hier){
    int size = lvl.matrix.size();
    std::vector<double> phi(size,1000);
    std::vector<double> rhs(size,-1000);
    std::cout<<"AMG level="<<lvl.level<<"   ";
    sor(lvl.matrix,rhs,phi,maxerror,maxsteps,omega);
  }

  for(auto& lvl:hier){
    int size = lvl.matrix.size();
    std::vector<double> phi(size,1000);
    std::vector<double> rhs(size,-1000);
    std::cout<<"AMG level="<<lvl.level<<"   ";
    jacobi(lvl.matrix,rhs,phi,maxerror,maxsteps,omega);
  }
}
#endif


//v-cycle: move straight down to coarsest level, 
//then back up and repeat until convergence
void solve_v_cycle(AMGhierarchy& Hier, const std::vector<double>& rho, std::vector<double>& phi, int maxsteps, 
		   int smooth_steps=5, bool gen_flag=0, double maxerror=1e-12){

  int max_level=Hier[0].max_level;
  //int max_level=2;
  int steps=0;
  std::vector<int> stepsvec(max_level,smooth_steps);
  //if gen flag=true, use generalized cycle
  if(gen_flag){
    for(auto i=0;i<max_level;++i){
      stepsvec[i] = (i+1) * smooth_steps;
    }
  }

  AMGlvl& coarsest_lvl = Hier[max_level-1];

  //init first guess for each level
  Hier[0].phi = phi;
  Hier[0].rhs = rho;
  for(auto i=1;i<max_level;++i){
    std::fill(Hier[i].phi.begin(),Hier[i].phi.end(),0);
    std::fill(Hier[i].rhs.begin(),Hier[i].rhs.end(),0);
  }

  auto v_iteration = [&](){
    double error = 0;

    //go "down" from i to next coarser lvl all the way to coarsest level
    for(auto i=0;i<max_level-1;++i){ 
      double cycerror = go_down(i,Hier,maxerror,stepsvec[i]);
      if(i==0){
	error=cycerror;
	if(error < maxerror) return error;
      }
    }
    
    //almost exact solution on coarsest grid
    cg(coarsest_lvl.matrix,coarsest_lvl.rhs,coarsest_lvl.phi,std::numeric_limits<int>::max());

    //go back "up" again from i t0 next finer lvl
    for(auto i=max_level-1;i>0;--i) go_up(i,Hier,maxerror,stepsvec[i]);

    return error;
  };

  double error=1;
  do{
    error = v_iteration();
    ++steps;
    std::cout<<"v-cycle step "<<steps<<", error "<<error<<std::endl;
  }while(error>maxerror && steps<maxsteps);

  phi=Hier[0].phi;
  std::cout<<"AMG V-cycle: cycles="<<steps<<"  "<<"error="<<error<<std::endl;
}


//TODO pass parameters of amg
void amg_preconditioned_cg(AMGhierarchy& hier, const std::vector<double>& rhs, std::vector<double>& phi, 
			   std::function<void(AMGhierarchy&,const std::vector<double>&,std::vector<double>&,int,int,bool,double)> precondition, 
			   bool gen_flag, int maxsteps, double acc=1e-12){

  //system matrix the same for both systems
  Matrix& matrix = hier[0].matrix;
  assert(matrix.size() == rhs.size() && "matrix & vector size dont match in cg(...)");
  assert(matrix.size() == phi.size() && "matrix & vector size dont match in cg(...)");

  double error=0;
  int steps=0;

  //move out of here and cg() from above
  auto scalar_product = [](const std::vector<double>& v1, const std::vector<double>& v2){
    assert(v1.size() == v2.size() && "vector dimensions dont match in scalar_product");
    double result=0;
    for(auto i=0u;i<v1.size();++i) result+=v1[i]*v2[i];
    return result;
  };

  //needed to set residual as rhs for preconditioner
  std::vector<double> residual=matrix_vector(matrix,phi);
  for(auto i=0u;i<residual.size();++i) residual[i] = rhs[i] - residual[i];

  //preconditioned variable used here
  std::vector<double> prec_residual(residual);
  precondition(hier,residual,prec_residual,30,2,gen_flag,1e-12);
  //search direction in preconditioned residual
  std::vector<double> direction(prec_residual);
  std::vector<double> mvprod(direction.size(),0);

  double delta = scalar_product(residual,prec_residual);
  double new_delta = delta;


  for(auto i=0;i<maxsteps;++i){
    error=0;
    ++steps;

    //save this mvproduct
    mvprod = matrix_vector(matrix,direction);

    //find phi in search direction and update residual
    double alpha = delta;
    alpha /= scalar_product(direction,mvprod);

    for(auto k=0u;k<phi.size();++k) phi[k]+=alpha*direction[k];
    for(auto k=0u;k<residual.size();++k){ 
      residual[k] -= alpha*mvprod[k];
      if(fabs(residual[k]) > error) error = fabs(residual[k]);
    }
    if(error < acc) break;

    //use preconditioning
    precondition(hier,residual,prec_residual,30,2,gen_flag,1e-12);
    
    //update coefficients for new search direction
    new_delta = scalar_product(residual,prec_residual);
    double beta = new_delta / delta;
    delta = new_delta;

    //std::cout<<"step="<<steps<<"  error="<<error<<"  alpha="<<alpha<<"  beta="<<beta<<"  delta="<<delta<<std::endl;

    //update search direction
    for(auto k=0u;k<direction.size();++k) direction[k] = prec_residual[k] + beta*direction[k];
  }

  cout<<"amg_preconditioned_cg: error="<<error<<"  steps="<<steps<<endl;
}




//TODO XXX FIXME whats still to do:
//CLEAN UP
//implement different cycles
//nice to have: automatic benchmarking to find fastest method
int main(){
  Matrix matrix;
  vector<double> dens;

  matrix_setup(matrix,NR,NZ,0.5);
  fill_rhs(dens,NR,NZ);
  AMGhierarchy AMG = create_amg_hierarchy(matrix,16,40,0.25,0.35);
  print_hierarchy(AMG,NR,NZ,"hierarchy.dat");
  std::vector<double> phi(matrix.size(),1);

  solve_v_cycle(AMG,dens,phi,1000,5,0,1e-12);
  std::pair<int,double> verror = calc_max_norm_res(matrix,dens,phi);
  std::fill(phi.begin(),phi.end(),0);
  solve_v_cycle(AMG,dens,phi,1000,2,1,1e-12);
  std::pair<int,double> vgenerror = calc_max_norm_res(matrix,dens,phi);
  std::fill(phi.begin(),phi.end(),0);
  //amg_preconditioned_cg(AMG,dens,phi,solve_v_cycle,0,1000);
  std::pair<int,double> pcgerror = calc_max_norm_res(matrix,dens,phi);
  std::fill(phi.begin(),phi.end(),0);
  //amg_preconditioned_cg(AMG,dens,phi,solve_v_cycle,1,1000);
  std::pair<int,double> gen_pcgerror = calc_max_norm_res(matrix,dens,phi);

  std::cout<<"    vcycle error is "<<verror.second<<" at "<<verror.first<<std::endl;
  std::cout<<"gen vcycle error is "<<vgenerror.second<<" at "<<vgenerror.first<<std::endl;
  std::cout<<"       pcg error is "<<pcgerror.second<<" at "<<pcgerror.first<<std::endl;
  std::cout<<"   gen_pcg error is "<<gen_pcgerror.second<<" at "<<gen_pcgerror.first<<std::endl;

  return 0;
}

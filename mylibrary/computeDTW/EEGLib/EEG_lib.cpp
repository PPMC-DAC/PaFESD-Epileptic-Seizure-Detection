#include "EEG_lib.hpp"
#include <cmath>

// Returns two versions of the statiscs: one with mu and sigma interleaved and another with mu and sigma separated for SIMD version
std::pair<FloatVector,FloatVector> computeStatistics(const FloatVector& V, const size_t numEpochs, const size_t nsamplesEpoch, const size_t strideEpoch){
  
  FloatVector statistics, statisticsSIMD(2*numEpochs);
  statistics.reserve(2 * numEpochs);

  for (size_t i = 0; i < numEpochs; ++i) {
    float sum = 0.0;
    float sumSquares = 0.0;

    for (size_t j = i * strideEpoch; j < i * strideEpoch + nsamplesEpoch; ++j) {
        sum += V[j];
        sumSquares += V[j] * V[j];
    }

    float avg = sum / nsamplesEpoch;
    //we store the inverse of sigma= 1/m * sum(x_i - mu)^2 = 1/m * sum(x_i^2) - mu^2
    float invdev = 1 / std::sqrt((sumSquares / nsamplesEpoch) - (avg * avg));
    statistics.push_back(avg);
    statistics.push_back(invdev);
    statisticsSIMD[i]=avg;
    statisticsSIMD[i+numEpochs]=invdev;
  }

  return std::pair{statistics,statisticsSIMD};
}

/// Calculate Dynamic Time Warpping distance
/// A,B: data and query, respectively, already z-normalized
/// r  : size of Sakoe-Chiba warpping band
/// workSpace : prealocated space of size 4*r+2 to accomodate 2 warpping bands
#define dist_sqrt(x,y) (sqrt((y-x)*(y-x)))
#define dist(x,y) ((y-x)*(y-x))
#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))
data_t single_dtw(const data_t* A, const data_t* B, const size_t m, const int r, data_t * workSpace)
{
    int k=0;
    data_t x,y,z;
    //workSpace should have space for 4*r+2!! Each band occupies 2*r+1
    data_t *cost=workSpace;                 //Workspace for the DTW: current band/row
    data_t *cost_prev=workSpace+(2*r)+1;    //Workspace for the DTW: previous band/row
    data_t *cost_tmp;

    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
    //for(int k=0; k<(4*r)+2; k++) workSpace[k]=maxval;   //Just in case the compiler does not optimize the loop
    memset(workSpace, maxval, (4*r+2)*sizeof(data_t));    //let's call memset directly. sizeof(workSpace) does not work

    for (int i=0; i<m; i++) //Traverses the rows
    {
        k = max(0,r-i); // r, r-1, r-2, ... 0 Take care with size_t r -> r-i always >0 in the comparison
        for(int j=max(0,i-r); j<=min(m-1,i+r); j++, k++) //Traverses the band: 2*r+1 iterations in the worst case
        {
            /// Initialize the first element of the cost matrix
            if ((i==0)&&(j==0))
            {
                cost[k]=dist(A[0],B[0]);
                continue;
            }
            if ((j-1<0)||(k-1<0))     y = maxval;
            else                      y = cost[k-1];
            if ((i-1<0)||(k+1>2*r))   x = maxval;
            else                      x = cost_prev[k+1];
            if ((i-1<0)||(j-1<0))     z = maxval;
            else                      z = cost_prev[k];

            /// Classic DTW calculation
            cost[k] = min(min(x, y), z) + dist(A[i],B[j]);
        }
        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;
    /// the DTW distance is at the middle of our band array.
    return cost_prev[k];
}

FloatVector DTWCPU(const FloatVector &S, const FloatVector &Q, const FloatVector &sttS, const FloatVector &sttQ, 
                   const blockRange blk, const size_t sEpoPat, const size_t stride, const size_t w, const int verbose)
{
  auto start = std::chrono::high_resolution_clock::now();
  
  auto regPat = std::make_unique<data_t[]>(blk.nPat*sEpoPat); //bank of z-normalized patterns
  
  //tbb::parallel_for((int)blk.pId, (int)(blk.pId+blk.nPat), [&](size_t patID){   //In parallel: one iteration per pattern    
#pragma omp parallel for shared(regPat)  
  for (int patID=blk.pId; patID<(blk.pId + blk.nPat); patID++){      //inizialization of the z-normalized bank of patterns 
    data_t meanPat=sttQ[patID*2];              //Mean of the actual pattern  
    data_t invSigmaPat=sttQ[patID*2+1];    //Inverse of the deviation of the actual pattern
    for (int j=0; j<sEpoPat; j++){
      //note that regPat is relative to 0, therefore the m-blk.pId
      regPat[(patID-blk.pId)*sEpoPat+j]=(Q[patID*stride+j]-meanPat)*invSigmaPat;   //values are stored after Z-normalization 
    }
  }
  //);

  FloatVector dtw_res(blk.nPat * blk.nEpo); //Vector of dtw's returned. One result for each Epoch and Pattern
  // tbb::parallel_for(tbb::blocked_range<int>{(int)blk.eId, (int)(blk.eId+blk.nEpo)}, 
  //                   [&](const tbb::blocked_range<int>& r){   //In parallel: one iteration per each Epoch
    //for (int epID=r.begin(); epID<r.end(); epID++){ //epoch loop
#pragma omp parallel for shared(dtw_res)
    for (int epID=blk.eId; epID<(blk.eId+blk.nEpo); epID++){ //epoch loop
      auto Epoch = std::make_unique<data_t[]>(sEpoPat); //Private epoch for each task
      data_t workSpace[4*w+2];                       //Private workSpace for each task
      data_t meanEp=sttS[epID*2];                    //Mean of the current Epoch  
      data_t invSigmaEp=sttS[epID*2+1];              //Deviation of the current Epoch
      size_t ptr=epID*stride;                        //Pointer to the Epoch base index
      for(int k=0; k<sEpoPat; k++){                     //Z-normalize the Epoch
        Epoch[k]=(S[ptr+k]-meanEp)*invSigmaEp;
      }
      for(int patID=0; patID<blk.nPat; patID++)       //For each pattern in the bank
        //note that dtw_res is relative to 0, therefore the epID-blk.eId
        // epoch wise:
        //dtw_res[patID*blk.nEpo+(epID-blk.eId)]=single_dtw(&Epoch[0], &regPat[patID*sEpoPat], sEpo, w, workSpace);
        // pattern wise
        dtw_res[(epID-blk.eId)*blk.nPat+patID]=single_dtw(&Epoch[0], &regPat[patID*sEpoPat], sEpoPat, w, workSpace);
    } // epoch loop
  //}
  //); //parallel_for   
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration<double,std::milli>(end - start).count();
  if(verbose)
    std::cout << "Time CPU: "<< time << " ms. for block: " << blk.eId << " "  << blk.nEpo << " " << blk.pId << " " << blk.nPat 
            << "; " << time/(blk.nPat * blk.nEpo) << "msec. per DTW\n"; 

  // std::cout <<" CPU DTWs\n";
  // for(int p=0; p<blk.nPat; p++){
  //   for(int e=0; e<blk.nEpo; e++){
  //     printf("dtw[%d,%d]=%f ",p,e,dtw_res[p*blk.nEpo+e]);
  //   }
  //   std::cout <<"\n";
  // }
  return dtw_res;
}

FloatVector get_distance_matrix(const data_t* S, const size_t nS, const data_t* Q, const size_t nQ, 
                                const size_t sEpoPat, const size_t stride, const size_t w, const int verbose) {

  size_t nEpo = (nS-sEpoPat)/stride+1;   // integer division
  size_t nPat = (nQ-sEpoPat)/stride+1;   // integer division

  if(verbose){
    std::cout << "Requested number of epochs: " << nEpo << " and patterns: " << nPat << "\n";
    setlocale(LC_NUMERIC, ""); //using comma as thousands separator in C
    printf("Processing %lu epochs x %lu patterns (#samples in S=%'lu, #samples in Q=%'lu)\n\n", nEpo, nPat, nS, nQ);
  }
  assert(nS > sEpoPat && nQ > sEpoPat);        // The signal should be larger than a single epoch

  // Initialize Signal and Query samples  
  FloatVector sig_S(nS);    // Signal
  FloatVector sig_Q(nQ);    // Query

  auto t_start = std::chrono::high_resolution_clock::now();

  std::copy(S, S+nS, sig_S.begin());
  std::copy(Q, Q+nQ, sig_Q.begin());
  
  //for(int p=0; p<100; p++) std::cout << sig_S[p] << "  " << sig_Q[p] << "\n" ;  //a look into the signal and the query
  auto t_end = std::chrono::high_resolution_clock::now();
  if(verbose)
    std::cout << "Time to initialize signals: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n";

  t_start = std::chrono::high_resolution_clock::now();
  FloatVector sttts_S, sttts_Q, sttts_SSIMD, sttts_QSIMD; // Statistics of S and Q 
  tie(sttts_S,sttts_SSIMD)=computeStatistics(sig_S, nEpo, sEpoPat, stride); // Include the mean and 1/desviation for each Epoch 
  tie(sttts_Q,sttts_QSIMD)=computeStatistics(sig_Q, nPat, sEpoPat, stride); // Include the mean and desviation for each Epoch 
  t_end = std::chrono::high_resolution_clock::now();
  if(verbose)
    std::cout << "Time to initialize statistics: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n\n";

  FloatVector dtw_res;
  // the blockRange specifies the range of epochs and patterns to be processed
  blockRange blk{0, nEpo, 0, nPat};
  t_start = std::chrono::high_resolution_clock::now();
  dtw_res=DTWCPU(sig_S, sig_Q, sttts_S, sttts_Q, blk, sEpoPat, stride, w, verbose);
  t_end = std::chrono::high_resolution_clock::now();
  if(verbose)
    std::cout << "Time to compute "<< nEpo*nPat <<" DTWs: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n\n";

  return dtw_res;
}

// This function computes the DTW as in the dtw-python library (full euclidean distance with sqrt and symmetric2 pattern)
data_t single_dtw_sqrt_sym2(const data_t* A, const data_t* B, const size_t m, const int r, data_t * workSpace)
{
    int k=0;
    data_t x,y,z;
    //workSpace should have space for 4*r+2!! Each band occupies 2*r+1
    data_t *cost=workSpace;                 //Workspace for the DTW: current band/row
    data_t *cost_prev=workSpace+(2*r)+1;    //Workspace for the DTW: previous band/row
    data_t *cost_tmp;

    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
    //for(int k=0; k<(4*r)+2; k++) workSpace[k]=maxval;   //Just in case the compiler does not optimize the loop
    memset(workSpace, maxval, (4*r+2)*sizeof(data_t));    //let's call memset directly. sizeof(workSpace) does not work

    for (int i=0; i<m; i++) //Traverses the rows
    {
        k = max(0,r-i); // r, r-1, r-2, ... 0 Take care with size_t r -> r-i always >0 in the comparison
        for(int j=max(0,i-r); j<=min(m-1,i+r); j++, k++) //Traverses the band: 2*r+1 iterations in the worst case
        {
            /// Initialize the first element of the cost matrix
            data_t current_dist=dist_sqrt(A[i],B[j]);
            if ((i==0)&&(j==0))
            {
                cost[k]=current_dist;
                continue;
            }
            if ((j-1<0)||(k-1<0))     y = maxval;
            else                      y = cost[k-1]+current_dist;
            if ((i-1<0)||(k+1>2*r))   x = maxval;
            else                      x = cost_prev[k+1]+current_dist;
            if ((i-1<0)||(j-1<0))     z = maxval;
            else                      z = cost_prev[k]+2*current_dist;

            /// Classic DTW calculation
            cost[k] = min(min(x, y), z);
        }
        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;
    /// the DTW distance is at the middle of our band array.
    return cost_prev[k];
}



FloatVector DTWCPU_u(const data_t S[], const data_t Q[], const blockRange blk, 
                   const size_t sEpoPat, const size_t stride, const size_t w, const int verbose)
{
  auto start = std::chrono::high_resolution_clock::now();

  FloatVector dtw_res(blk.nPat * blk.nEpo); //Vector of dtw's returned. One result for each Epoch and Pattern

#pragma omp parallel for shared(S,Q,dtw_res)
    for (int epID=blk.eId; epID<(blk.eId+blk.nEpo); epID++){ //epoch loop
      data_t workSpace[4*w+2];                       //Private workSpace for each task
      size_t ptr=epID*stride;                        //Pointer to the Epoch base index
      for(int patID=0; patID<blk.nPat; patID++)       //For each pattern in the bank
        //note that dtw_res is relative to 0, therefore the epID-blk.eId
        dtw_res[(epID-blk.eId)*blk.nPat+patID]=single_dtw_sqrt_sym2(&S[ptr], &Q[patID*stride], sEpoPat, w, workSpace);
    } // epoch loop
  
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration<double,std::milli>(end - start).count();
  if(verbose)
    std::cout << "Time CPU: "<< time << " ms. for block: " << blk.eId << " "  << blk.nEpo << " " << blk.pId << " " << blk.nPat 
            << "; " << time/(blk.nPat * blk.nEpo) << "msec. per DTW\n"; 

  return dtw_res;
}

FloatVector get_distance_matrix_u(const data_t* S, const size_t nS, const data_t* Q, const size_t nQ, 
                                const size_t sEpoPat, const size_t stride, const size_t w, const int verbose) {

  size_t nEpo = (nS-sEpoPat)/stride+1;   // integer division
  size_t nPat = (nQ-sEpoPat)/stride+1;   // integer division

  if(verbose){
    std::cout << "Requested number of epochs: " << nEpo << " and patterns: " << nPat << "\n";
    setlocale(LC_NUMERIC, ""); //using comma as thousands separator in C
    printf("Processing %lu epochs x %lu patterns (#samples in S=%'lu, #samples in Q=%'lu)\n\n", nEpo, nPat, nS, nQ);
  }
  assert(nS > sEpoPat && nQ > sEpoPat);        // The signal should be larger than a single epoch
  //for(int p=0; p<100; p++) std::cout << sig_S[p] << "  " << sig_Q[p] << "\n" ;  //a look into the signal and the query
  
  FloatVector dtw_res;
  // the blockRange specifies the range of epochs and patterns to be processed
  blockRange blk{0, nEpo, 0, nPat};
  auto t_start = std::chrono::high_resolution_clock::now();
  dtw_res=DTWCPU_u(S, Q, blk, sEpoPat, stride, w, verbose);
  auto t_end = std::chrono::high_resolution_clock::now();
  if(verbose)
    std::cout << "Time to compute "<< nEpo*nPat <<" DTWs: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n\n";

  return dtw_res;
}

// This function computes the DTW similarly to the dtw-python library with symmetric 2 pattern but partial euclidean distance without sqrt
data_t single_dtw_sym2(const data_t* A, const data_t* B, const size_t m, const int r, data_t * workSpace)
{
    int k=0;
    data_t x,y,z;
    //workSpace should have space for 4*r+2!! Each band occupies 2*r+1
    data_t *cost=workSpace;                 //Workspace for the DTW: current band/row
    data_t *cost_prev=workSpace+(2*r)+1;    //Workspace for the DTW: previous band/row
    data_t *cost_tmp;

    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
    //for(int k=0; k<(4*r)+2; k++) workSpace[k]=maxval;   //Just in case the compiler does not optimize the loop
    memset(workSpace, maxval, (4*r+2)*sizeof(data_t));    //let's call memset directly. sizeof(workSpace) does not work

    for (int i=0; i<m; i++) //Traverses the rows
    {
        k = max(0,r-i); // r, r-1, r-2, ... 0 Take care with size_t r -> r-i always >0 in the comparison
        for(int j=max(0,i-r); j<=min(m-1,i+r); j++, k++) //Traverses the band: 2*r+1 iterations in the worst case
        {
            /// Initialize the first element of the cost matrix
            data_t current_dist=dist(A[i],B[j]);
            if ((i==0)&&(j==0))
            {
                cost[k]=current_dist;
                continue;
            }
            if ((j-1<0)||(k-1<0))     y = maxval;
            else                      y = cost[k-1]+current_dist;
            if ((i-1<0)||(k+1>2*r))   x = maxval;
            else                      x = cost_prev[k+1]+current_dist;
            if ((i-1<0)||(j-1<0))     z = maxval;
            else                      z = cost_prev[k]+2*current_dist;

            /// Classic DTW calculation
            cost[k] = min(min(x, y), z);
        }
        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;
    /// the DTW distance is at the middle of our band array.
    return cost_prev[k];
}
// _u stands for un-z-normalized and _ns for no sqrt
FloatVector DTWCPU_u_ns(const data_t S[], const data_t Q[], const blockRange blk, 
                   const size_t sEpoPat, const size_t stride, const size_t w, const int verbose)
{
  auto start = std::chrono::high_resolution_clock::now();

  FloatVector dtw_res(blk.nPat * blk.nEpo); //Vector of dtw's returned. One result for each Epoch and Pattern

#pragma omp parallel for shared(S,Q,dtw_res)
    for (int epID=blk.eId; epID<(blk.eId+blk.nEpo); epID++){ //epoch loop
      data_t workSpace[4*w+2];                       //Private workSpace for each task
      size_t ptr=epID*stride;                        //Pointer to the Epoch base index
      for(int patID=0; patID<blk.nPat; patID++)       //For each pattern in the bank
        //note that dtw_res is relative to 0, therefore the epID-blk.eId
        dtw_res[(epID-blk.eId)*blk.nPat+patID]=single_dtw_sym2(&S[ptr], &Q[patID*stride], sEpoPat, w, workSpace);
    } // epoch loop
  
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration<double,std::milli>(end - start).count();
  if(verbose)
    std::cout << "Time CPU: "<< time << " ms. for block: " << blk.eId << " "  << blk.nEpo << " " << blk.pId << " " << blk.nPat 
            << "; " << time/(blk.nPat * blk.nEpo) << "msec. per DTW\n"; 

  return dtw_res;
}

FloatVector get_distance_matrix_u_ns(const data_t* S, const size_t nS, const data_t* Q, const size_t nQ, 
                                const size_t sEpoPat, const size_t stride, const size_t w, const int verbose) {

  size_t nEpo = (nS-sEpoPat)/stride+1;   // integer division
  size_t nPat = (nQ-sEpoPat)/stride+1;   // integer division

  if(verbose){
    std::cout << "Requested number of epochs: " << nEpo << " and patterns: " << nPat << "\n";
    setlocale(LC_NUMERIC, ""); //using comma as thousands separator in C
    printf("Processing %lu epochs x %lu patterns (#samples in S=%'lu, #samples in Q=%'lu)\n\n", nEpo, nPat, nS, nQ);
  }
  assert(nS > sEpoPat && nQ > sEpoPat);        // The signal should be larger than a single epoch
  //for(int p=0; p<100; p++) std::cout << sig_S[p] << "  " << sig_Q[p] << "\n" ;  //a look into the signal and the query
  
  FloatVector dtw_res;
  // the blockRange specifies the range of epochs and patterns to be processed
  blockRange blk{0, nEpo, 0, nPat};
  auto t_start = std::chrono::high_resolution_clock::now();
  dtw_res=DTWCPU_u_ns(S, Q, blk, sEpoPat, stride, w, verbose);
  auto t_end = std::chrono::high_resolution_clock::now();
  if(verbose)
    std::cout << "Time to compute "<< nEpo*nPat <<" DTWs: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n\n";

  return dtw_res;
}
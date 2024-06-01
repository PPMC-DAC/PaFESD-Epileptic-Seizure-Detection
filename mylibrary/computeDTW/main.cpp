#include <locale.h>
#include "EEG_lib.hpp"
#include <tbb/tbb.h>

// n stands for number (number of patterns, epochs, ...), s for size in samples 
constexpr size_t sPat=1280;       // size of patterns
constexpr size_t sEpo=sPat;       //size of Epochs
constexpr size_t stride=256;      //samples between consecutive patterns and epochs (sliding window stride) 

constexpr size_t w=16;           //warping window size of the Sakoe-Chiba band
constexpr size_t sBand=(2*w)+1;  //full size of the Sakoe-Chiba band (warping window)

int main(int argc, char** argv) {

  size_t nS = (argc>1) ? atol(argv[1]) : 300000; //Number of samples in the signal S
  size_t nQ = (argc>2) ? atol(argv[2]) : 10000;  //Number of samples in the query Q
  size_t nEpo = (nS-sEpo)/stride+1;   // integer division
  size_t nPat = (nQ-sPat)/stride+1;   // integer division

  std::cout << "Requested number of epochs: " << nEpo << " and patterns: " << nPat << "\n";
  //But padding is exercised in order to have a number of patters multiple SIMD width
  size_t SIMDw=simd_t::size();
  nPat = ((nPat-1)/SIMDw+1)*SIMDw; // nearest multplie of SIMDw
  nQ=sPat+stride*(nPat-1);
  
  printf("But padding is applied considering that SIMD width=%d.\n", SIMDw);
  setlocale(LC_NUMERIC, ""); //using comma as thousands separator in C
  printf("After padding we end up processing %lu epochs x %lu patterns (#samples in S=%'lu, #samples in Q=%'lu)\n\n", nEpo, nPat, nS, nQ);


  setlocale(LC_NUMERIC, ""); //using comma as thousands separator in C
  printf("Processing %lu epochs x %lu patterns (#samples in S=%'lu, #samples in Q=%'lu)\n\n", nEpo, nPat, nS, nQ);
  assert(nS > sEpo && nQ > sPat);        // The signal should be larger than a single epoch

  // Initialize random number generator
  std::random_device seed;    // Random device seed
  std::mt19937 mte{seed()};   // mersenne_twister_engine
  std::uniform_real_distribution<> uniform{-0.5, 0.5};
  // Initialize Signal and Query samples  
  FloatVector sig_S;    // Signal
  sig_S.reserve(nS);    // Signal vector prealocated
  FloatVector sig_Q;    // Query
  sig_Q.reserve(nQ);    // Query vector prealocated

  auto t_start = std::chrono::high_resolution_clock::now();
  tbb::parallel_invoke( //Generate S and Q in parallel
    [&](){
        std::generate_n(std::back_inserter(sig_S), nS, [&] { return uniform(mte); }); //Init with random
        std::partial_sum(sig_S.begin(), sig_S.end(), sig_S.begin()); // Build the random-walk time series
    },
    [&](){
        std::generate_n(std::back_inserter(sig_Q), nQ, [&] { return uniform(mte); }); //Init with random
        std::partial_sum(sig_Q.begin(), sig_Q.end(), sig_Q.begin()); // Build the random-walk time series
    }
  );
  //for(int p=0; p<100; p++) std::cout << sig_S[p] << "  " << sig_Q[p] << "\n" ;  //a look into the signal and the query
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << "Time to initialize signals: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n";

  t_start = std::chrono::high_resolution_clock::now();
  FloatVector sttts_S, sttts_Q, sttts_SSIMD, sttts_QSIMD; // Statistics of S and Q
  tbb::parallel_invoke( //Compute statistics of S and Q in parallel
    [&](){
      tie(sttts_S,sttts_SSIMD)=computeStatistics(sig_S, nEpo, sEpo, stride); // Include the mean and 1/desviation for each Epoch 
    },
    [&](){
      tie(sttts_Q,sttts_QSIMD)=computeStatistics(sig_Q, nPat, sPat, stride); // Include the mean and desviation for each Epoch 
    }
  );
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << "Time to initialize statistics: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n\n";

  std::cout << "Computing DTW on CPU with SIMD width: "<< simd_t::size() << "\n";
  FloatVector dtw_res, dtw_gold;
  size_t error=0;
  // the blockRange specifies the range of epochs and patterns to be processed
  blockRange blk{0, nEpo, 0, nPat};
  t_start = std::chrono::high_resolution_clock::now();
  tbb::parallel_invoke( //Compute statistics of S and Q in parallel
    [&](){dtw_res=DTWCPUSIMD(sig_S, sig_Q, sttts_SSIMD, nEpo, sttts_QSIMD, nPat, blk);},
    [&](){dtw_gold=DTWCPU(sig_S, sig_Q, sttts_S, sttts_Q, blk);});
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << "Time to compute "<< nEpo*nPat <<" DTWs: "<< std::chrono::duration<double,std::milli>(t_end - t_start).count() << " ms.\n\n";

  //test_simdDTW();
  error+=checkResults(dtw_res, dtw_gold, blk);
  std::cout << "Total number of errors: " << error << "\n";
  return 0;
}           
            
           
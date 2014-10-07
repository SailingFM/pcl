#ifndef TRACKING_STRATIFIED_PARTICLE_FILTER_OMP_H_
#define TRACKING_STRATIFIED_PARTICLE_FILTER_OMP_H_

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/stratified_point_cloud_coherence.h>

namespace pcl
{
  namespace tracking
  {
    template <typename PointInT, typename StateT>
    class StratifiedParticleFilterOMPTracker: public ParticleFilterOMPTracker<PointInT, StateT>
    {
    public:
      using Tracker<PointInT, StateT>::tracker_name_;
      using Tracker<PointInT, StateT>::search_;
      using Tracker<PointInT, StateT>::input_;
      using Tracker<PointInT, StateT>::indices_;
      using Tracker<PointInT, StateT>::getClassName;
      using ParticleFilterTracker<PointInT, StateT>::particles_;
      using ParticleFilterTracker<PointInT, StateT>::change_detector_;
      using ParticleFilterTracker<PointInT, StateT>::change_counter_;
      using ParticleFilterTracker<PointInT, StateT>::change_detector_interval_;
      using ParticleFilterTracker<PointInT, StateT>::use_change_detector_;
      using ParticleFilterTracker<PointInT, StateT>::alpha_;
      using ParticleFilterTracker<PointInT, StateT>::changed_;
      using ParticleFilterTracker<PointInT, StateT>::use_normal_;
      using ParticleFilterTracker<PointInT, StateT>::particle_num_;
      using ParticleFilterTracker<PointInT, StateT>::change_detector_filter_;
      using ParticleFilterTracker<PointInT, StateT>::transed_reference_vector_;
      using ParticleFilterTracker<PointInT, StateT>::ref_;
      using ParticleFilterTracker<PointInT, StateT>::representative_state_;
      using ParticleFilterTracker<PointInT, StateT>::motion_;
      using ParticleFilterTracker<PointInT, StateT>::motion_ratio_;
      using ParticleFilterTracker<PointInT, StateT>::initial_noise_mean_;
      using ParticleFilterTracker<PointInT, StateT>::initial_noise_covariance_;
      using ParticleFilterTracker<PointInT, StateT>::step_noise_covariance_;
      using ParticleFilterTracker<PointInT, StateT>::trans_;
      using ParticleFilterTracker<PointInT, StateT>::resample_likelihood_thr_;
      using ParticleFilterTracker<PointInT, StateT>::iteration_num_;
      using ParticleFilterTracker<PointInT, StateT>::normalizeWeight;
      using ParticleFilterTracker<PointInT, StateT>::normalizeParticleWeight;
      using ParticleFilterTracker<PointInT, StateT>::calcBoundingBox;
      using ParticleFilterTracker<PointInT, StateT>::initParticles;
      using ParticleFilterTracker<PointInT, StateT>::toEigenMatrix;
      using ParticleFilterTracker<PointInT, StateT>::getResult;
      using ParticleFilterTracker<PointInT, StateT>::sampleWithReplacement;
      using ParticleFilterTracker<PointInT, StateT>::setInputCloud;
      
      typedef Tracker<PointInT, StateT> BaseClass;

      typedef PointCloud<PointInT> CloudInT;
      typedef boost::shared_ptr< CloudInT > CloudInTPtr;
      typedef boost::shared_ptr< const CloudInT > CloudInTConstPtr;
      
      typedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
      typedef typename PointCloudState::Ptr PointCloudStatePtr;
      typedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;

      typedef PointCoherence<PointInT> Coherence;
      typedef boost::shared_ptr< Coherence > CoherencePtr;
      typedef boost::shared_ptr< const Coherence > CoherenceConstPtr;

      typedef PointCloudCoherence<PointInT> CloudCoherence;
      typedef boost::shared_ptr< CloudCoherence > CloudCoherencePtr;
      typedef boost::shared_ptr< const CloudCoherence > CloudCoherenceConstPtr;

      typedef pcl::tracking::StratifiedPointCloudCoherence<PointInT> StratCoherenceT;
      typedef boost::shared_ptr<StratCoherenceT> StratCoherencePtr;
      typedef boost::shared_ptr< const CloudCoherence > StratCoherenceConstPtr;
           
      /** \brief Initialize the scheduler and set the number of threads to use.
        * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        */      
      StratifiedParticleFilterOMPTracker (unsigned int nr_threads = 0)
      : ParticleFilterOMPTracker<PointInT, StateT> ()
      , threads_ (nr_threads)
      , last_stamp_ (0)
      , delta_t_ (1)
      {
        tracker_name_ = "StratifiedParticleFilterOMPTracker";
        T1 = T2 = T3 = 0;
        frame_count = 0;
        changed_ = true;
      }

      /** \brief Initialize the scheduler and set the number of threads to use.
        * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        */
      inline void
      setNumberOfThreads (unsigned int nr_threads = 0) { threads_ = nr_threads; }

      void 
      setInputCloud (const CloudInTConstPtr &cloud, double t_stamp);
        
      void printParticleWeights ()
      {
        for (int i = 0; i < particle_num_; ++i)
          std::cout << particles_->points[i].weight<<"\n";
      }
      
      bool 
      initCompute ();
      
      /** \brief THIS DESTROYS THE INPUT VECTOR, Filter takes ownership!!! */
      void
      setCloudCoherenceVector (std::vector<StratCoherencePtr> &coherence_vec)
      {
        coherences_.swap (coherence_vec);
      }

      void 
      updateStrata (std::map<uint32_t,typename SequentialSV::Ptr> &supervoxel_clusters);
      
    protected:
      void
      applyWeights ();
      
      void
      computeTracking ();
      
      /** \brief The number of threads the scheduler should use. */
      unsigned int threads_;

      /** \brief weighting phase of particle filter method.
          calculate the likelihood of all of the particles and set the weights.
        */
      virtual void weight ();
      
      void 
      update ();

      virtual void 
      resample ();

      double T1, T2, T3;
      int frame_count;
      double last_stamp_, delta_t_;
      
      std::vector<StratCoherencePtr> coherences_;
    };
  }
}
#ifdef PCL_NO_PRECOMPILE
#include <pcl/tracking/impl/stratified_particle_filter_omp.hpp>
#endif

#endif //TRACKING_STRATIFIED_PARTICLE_FILTER_OMP_H_


#ifndef PCL_TRACKING_STRATIFIED_POINT_CLOUD_COHERENCE_H_
#define PCL_TRACKING_STRATIFIED_POINT_CLOUD_COHERENCE_H_

#include <pcl/search/search.h>
#include <pcl/search/octree.h>
#include <pcl/segmentation/sequential_supervoxel_clustering.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/boost.h>
#include <boost/graph/graph_concepts.hpp>

namespace pcl
{
  namespace tracking
  {
    /** \brief @b StratifiedPointCloudCoherence computes coherence between two pointclouds using the
          provided strata for random sampling.
      * \author Jeremie Papon
      * \ingroup tracking
      */
    template <typename PointInT>
    class StratifiedPointCloudCoherence: public NearestPairPointCloudCoherence<PointInT>
    {
    public:
      typedef typename NearestPairPointCloudCoherence<PointInT>::PointCoherencePtr PointCoherencePtr;
      typedef typename NearestPairPointCloudCoherence<PointInT>::PointCloudInConstPtr PointCloudInConstPtr;
      typedef boost::shared_ptr<pcl::search::Search<PointInT> > SearchPtr;
      typedef boost::shared_ptr<const pcl::search::Search<PointInT> > SearchConstPtr;
      
      typedef pcl::PointXYZL LabelPointT;
      typedef pcl::PointCloud<LabelPointT> LabelCloudT;
      
      using NearestPairPointCloudCoherence<PointInT>::search_;
      using NearestPairPointCloudCoherence<PointInT>::maximum_distance_;
      using NearestPairPointCloudCoherence<PointInT>::target_input_;
      using NearestPairPointCloudCoherence<PointInT>::point_coherences_;
      using NearestPairPointCloudCoherence<PointInT>::coherence_name_;
      using NearestPairPointCloudCoherence<PointInT>::new_target_;
      using NearestPairPointCloudCoherence<PointInT>::getClassName;
      
      /** \brief empty constructor */
      StratifiedPointCloudCoherence () : 
        NearestPairPointCloudCoherence<PointInT> (),
        num_samples_ (1),
        threads_ (1)
      {
        coherence_name_ = "StratifiedPointCloudCoherence";
        rng_.seed (static_cast<unsigned int>(std::time(0))+seed_inc_);
        seed_inc_ += 17;
      }
      /** \brief compute coherence between two pointclouds after applying trans to points in cloud */
      inline void
      compute (const PointCloudInConstPtr &cloud, const Eigen::Affine3f &trans, float &w_j);
      
      inline bool
      initCompute ();
      
      void
      setStrata (const std::vector<pcl::SequentialSV::Ptr> &supervoxels);
      
      void 
      setNumSamplesPerStratum (int num_samples)
      {
        num_samples_ = num_samples;
        for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
        {
          strata_itr->sampled_.resize (num_samples);
        }
      }
      
      void
      setNumThreads (int num_threads)
      {
        threads_ = num_threads;
      }
      
      int 
      getNumStrata ()
      {
        return strata_indices_.size ();
      }

      void
      applyWeightToStrata (float weight);
      
      //! \brief Just used for debugging TODO REMOVE
      void
      printVoxelWeights ();
      
      void
      clearSupervoxelWeights ();
      
      void 
      normalizeSupervoxelWeights ();
      
      void 
      updateStrata (std::map<uint32_t,typename SequentialSV::Ptr> &supervoxel_clusters);
      
      void
      getSVLabels (std::vector<uint32_t> &sv_labels);
      
    protected:
      /** \brief compute the nearest pairs and compute coherence using point_coherences_ */
      virtual void
      computeCoherence (const PointCloudInConstPtr &cloud, const IndicesConstPtr &indices, float &w_j);
      
      void
      computeCoherence (const PointCloudInConstPtr &cloud,  const Eigen::Affine3f &trans, float &w);

      //! Number of Samples per stratum - default = 1
      int num_samples_;
      //! Random number generator for sampling
      boost::mt19937 rng_;
      //! Used to increment seed generator across instances of this class
      static int seed_inc_;
      
      int threads_;
      
      struct StratumHelper
      {
        StratumHelper (const SequentialSV::Ptr &supervoxel): 
          supervoxel_ (supervoxel),
          stratum_label_ (supervoxel->label_)
        { }
        
        StratumHelper (int stratum_label): 
          stratum_label_ (stratum_label)
        { }
        
        SequentialSV::Ptr supervoxel_;
        //! Indices of points in target_input_ which belong to this stratum
        std::pair<size_t,size_t> index_range_;
        //! Stratum Label
        uint32_t stratum_label_;

        std::vector<size_t> sampled_;
        
        bool operator< (const StratumHelper &r) const
        {
          return stratum_label_ < r.stratum_label_;
        }
      };

      typedef typename boost::ptr_list<StratumHelper>::iterator StrataItr;
      typedef std::pair<StrataItr, bool> StrataItrBoolPair;
      boost::ptr_list<StratumHelper> strata_indices_;
    };
  }
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/tracking/impl/stratified_point_cloud_coherence.hpp>
#endif

#endif //PCL_TRACKING_STRATIFIED_POINT_CLOUD_COHERENCE_H_


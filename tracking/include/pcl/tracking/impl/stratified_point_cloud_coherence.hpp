#ifndef PCL_TRACKING_IMPL_STRATIFIED_POINT_CLOUD_COHERENCE_H_
#define PCL_TRACKING_IMPL_STRATIFIED_POINT_CLOUD_COHERENCE_H_

#include <algorithm>

#include <pcl/search/octree.h>
#include <pcl/tracking/stratified_point_cloud_coherence.h>

namespace pcl
{
  namespace tracking
  {
    template <typename PointInT>
    int StratifiedPointCloudCoherence<PointInT>::seed_inc_ = 0;
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::setStrata (const std::vector<uint32_t> &strata_labels)
    {
      //Assign indices to appropriate strata
      for (size_t target_idx = 0; target_idx < strata_labels.size(); ++target_idx)
      {
        int stratum_label = strata_labels[target_idx];
        StrataItrBoolPair pair = strata_indices_.insert (new StratumHelper(stratum_label));
        pair.first->indices_.push_back (target_idx);
      }
      
      std::cout <<"Size of Strata:\n";
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
        std::cout <<strata_itr->stratum_label_<<"  has size "<<strata_itr->indices_.size ()<<"\n";
    }
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::setStrata (typename LabelCloudT::ConstPtr strata_label_cloud)
    {
      //Assign indices to appropriate strata
      for (size_t target_idx = 0; target_idx < strata_label_cloud->size(); ++target_idx)
      {
        int stratum_label = strata_label_cloud->points[target_idx].label;
        StrataItrBoolPair pair = strata_indices_.insert (new StratumHelper(stratum_label));
        pair.first->indices_.push_back (target_idx);
      }
      
      std::cout <<"Size of Strata:\n";
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
        std::cout <<strata_itr->stratum_label_<<"  has size "<<strata_itr->indices_.size ()<<"\n";
    }
    
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::computeCoherence (
        const PointCloudInConstPtr &cloud, const IndicesConstPtr &, float &w)
    {
      double val = 0.0;
      std::vector<int> k_indices(1);
      std::vector<float> k_distances(1);
      double max_dist_squared = maximum_distance_ * maximum_distance_;
      PointInT test_pt; 
      //std::cout << "====INDICES GENERATED:====\n";
      //Iterate through strata, drawing num_samples_ samples from each one uniformly.
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      {
        boost::uniform_int<int> index_dist(0, strata_itr->indices_.size ()-1);
        
        for (int i = 0; i < num_samples_; ++i)
        {
          int rnd_num = index_dist (rng_);
          //std::cout <<rnd_num<<"("<<strata_itr->indices_.size ()<<")"<<" ---> ";
          size_t idx = strata_itr->indices_[rnd_num];
          //std::cout <<idx<<"\n";
          copyPoint (cloud->points[idx], test_pt);
          //transformed_pt.getVector3fMap () = trans * cloud->points[idx].getVector3fMap ();
          search_->nearestKSearch (test_pt, 1, k_indices, k_distances);
          //TODO Change this is dumb dumb dumb - copy all the time?!?!
          PointInT target_point = target_input_->points[k_indices[0]];
          if (k_distances[0] < max_dist_squared)
          {
            double coherence_val = 1.0;
            for (size_t i = 0; i < point_coherences_.size (); i++)
            {
              PointCoherencePtr coherence = point_coherences_[i];  
              double w = coherence->compute (test_pt, target_point);
              coherence_val *= w;
            }
            val += coherence_val;
          }
        }
      }
      w = - static_cast<float> (val);
    }

    template <typename PointInT> bool
    StratifiedPointCloudCoherence<PointInT>::initCompute ()
    {
      if (!PointCloudCoherence<PointInT>::initCompute ())
      {
        PCL_ERROR ("[pcl::%s::initCompute] PointCloudCoherence::Init failed.\n", getClassName ().c_str ());
        return (false);
      }
      
      return true;
    }
    
  }
}

#define PCL_INSTANTIATE_StratifiedPointCloudCoherence(T) template class PCL_EXPORTS pcl::tracking::StratifiedPointCloudCoherence<T>;

#endif

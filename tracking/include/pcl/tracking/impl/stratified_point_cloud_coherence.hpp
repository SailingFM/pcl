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
    StratifiedPointCloudCoherence<PointInT>::setStrata (const std::vector<pcl::Supervoxel::Ptr> &supervoxels)
    {
      size_t idx_start = 0, idx_end;
      //Create the strata helpers
      strata_indices_.reserve (supervoxels.size ());
      for (int i = 0; i < supervoxels.size (); ++i)
      {
        idx_end = idx_start + supervoxels[i]->voxels_->size () - 1;
        strata_indices_.push_back (new StratumHelper(supervoxels[i]));
        //Assign indices
        strata_indices_.back ().index_range_ =std::make_pair<size_t,size_t> (idx_start,idx_end);
        strata_indices_.back ().sampled_.resize (num_samples_);
        idx_start = idx_end + 1;
      }
      //std::cout <<"Size of Strata:\n";
      //for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      //  std::cout <<strata_itr->stratum_label_<<"  has size "<<strata_itr->indices_.size ()<<"\n";
    }
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::applyWeightToStrata (float weight)
    {
      //#ifdef _OPENMP
      //#pragma omp parallel for num_threads(threads_) schedule(static, 10)
      //#endif
      for (int strata_idx = 0; strata_idx < strata_indices_.size (); ++strata_idx)
      {
        for (int i = 0; i < num_samples_; ++i)
        {
          // Add the voxel index and weight to this supervoxel
          strata_indices_[strata_idx].supervoxel_->voxel_weight_map_[strata_indices_[strata_idx].sampled_[i]] += weight;
        }
      }
    }
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::normalizeSupervoxelWeights ()
    {
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      {
        std::map <size_t, float>::iterator sv_map_itr = strata_itr->supervoxel_->voxel_weight_map_.begin ();
        for ( ; sv_map_itr != strata_itr->supervoxel_->voxel_weight_map_.end (); ++sv_map_itr)
        {
          // normalize by (Strata Size)/(Num Samples) - this makes expected value 
          //of each voxel 1 if all filters associated this voxel with this supervoxel 
          sv_map_itr->second *= static_cast<float> (strata_itr->supervoxel_->voxels_->size ()) / num_samples_;
        }
      }
    }
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::clearSupervoxelWeights ()
    {
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      {
        strata_itr->supervoxel_->voxel_weight_map_.clear ();
      }
    }
    
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::printVoxelWeights ()
    {
      std::cout <<"=========================================================\n";
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      {
        float sv_sum = 0;
        std::map <size_t, float>::iterator sv_map_itr = strata_itr->supervoxel_->voxel_weight_map_.begin ();
        std::cout << "SV="<<strata_itr->stratum_label_<<"   SV size="<<strata_itr->supervoxel_->voxels_->size () <<"  num voxels with nonzero weight="<<strata_itr->supervoxel_->voxel_weight_map_.size ()<<"\n";
        for ( ; sv_map_itr != strata_itr->supervoxel_->voxel_weight_map_.end (); ++sv_map_itr)
        {
          std::cout <<"idx="<<sv_map_itr->first<<"   w="<<sv_map_itr->second<<"\n";
          sv_sum += sv_map_itr->second;
        }
        
        std::cout <<"-------  sum = "<<sv_sum<<"-----------\n";
      }
      
    }
    
    

    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::computeCoherence (
      const PointCloudInConstPtr &cloud,  const Eigen::Affine3f &trans, float &w)
    {
      double val = 0.0;
      std::vector<int> k_indices(1);
      std::vector<float> k_distances(1);
      double max_dist_squared = maximum_distance_ * maximum_distance_;
      //Iterate through strata, drawing num_samples_ samples from each one uniformly.
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      {
        boost::uniform_int<int> index_dist(strata_itr->index_range_.first, strata_itr->index_range_.second);
        
        for (int i = 0; i < num_samples_; ++i)
        {
          int rnd_idx = index_dist (rng_);
          PointInT test_pt = cloud->points[rnd_idx];
          test_pt.getVector3fMap () = trans * test_pt.getVector3fMap ();
          search_->nearestKSearch (test_pt, 1, k_indices, k_distances);
          if (k_distances[0] < max_dist_squared)
          {
            double coherence_val = 1.0;
            for (size_t k = 0; k < point_coherences_.size (); k++)
            {
              PointCoherencePtr coherence = point_coherences_[k];
             // double w = coherence->compute (test_pt, target_input_->points[k_indices[0]]);
              double w = coherence->compute (test_pt, target_input_->points[k_indices[0]]);
              strata_itr->sampled_[i] = k_indices[0];
              coherence_val *= w;
            }
            val += coherence_val;
            
          }
          
        }
      }
      w = - static_cast<float> (val);
    }
    
    
    //
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::computeCoherence (
      const PointCloudInConstPtr &cloud, const IndicesConstPtr &, float &w)
    {
      /*
      double val = 0.0;
      std::vector<int> k_indices(1);
      std::vector<float> k_distances(1);
      double max_dist_squared = maximum_distance_ * maximum_distance_;
      PointInT test_pt; 
      //Iterate through strata, drawing num_samples_ samples from each one uniformly.
      for (StrataItr strata_itr = strata_indices_.begin (); strata_itr != strata_indices_.end (); ++strata_itr)
      {
        boost::uniform_int<int> index_dist(strata_itr->index_range_.first, strata_itr->index_range_.second);
        
        for (int i = 0; i < num_samples_; ++i)
        {
          int rnd_idx = index_dist (rng_);
          //std::cout <<rnd_num<<"("<<strata_itr->indices_.size ()<<")"<<" ---> ";
          //std::cout <<idx<<"\n";
          copyPoint (cloud->points[rnd_idx], test_pt);
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
     */ 
    }
    
    template <typename PointInT> void
    StratifiedPointCloudCoherence<PointInT>::compute (const PointCloudInConstPtr &cloud, const Eigen::Affine3f &trans, float &w)
    {
      if (!initCompute ())
      {
        PCL_ERROR ("[pcl::%s::compute] Init failed.\n", getClassName ().c_str ());
        return;
      }
      computeCoherence (cloud, trans, w);
    }
    
    template <typename PointInT> bool
    StratifiedPointCloudCoherence<PointInT>::initCompute ()
    {
      if (!target_input_ || target_input_->points.empty () || !search_ ||  target_input_ != search_->getInputCloud ())
      {
        if (!search_)
          PCL_ERROR ("[pcl::%s::compute] kd_tree is not built!!\n", getClassName ().c_str ());
        else if (!target_input_ || target_input_->points.empty ())
          PCL_ERROR ("[pcl::%s::compute] target_input_ is empty!\n", getClassName ().c_str ());
        else
          PCL_ERROR ("[pcl::%s::compute] kd_tree input cloud is not target_input!!!\n", getClassName ().c_str ());
        return false;
      }
      return true;
    }

  }
}

#define PCL_INSTANTIATE_StratifiedPointCloudCoherence(T) template class PCL_EXPORTS pcl::tracking::StratifiedPointCloudCoherence<T>;

#endif

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author : jpapon@gmail.com
 * Email  : jpapon@gmail.com
 *
 */

#ifndef PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_
#define PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_

#include <pcl/segmentation/sequential_supervoxel_clustering.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::SequentialSVClustering<PointT>::SequentialSVClustering (float voxel_resolution, float seed_resolution, bool use_single_camera_transform, bool prune_close_seeds) :
  SupervoxelClustering<PointT> (voxel_resolution, seed_resolution, use_single_camera_transform, prune_close_seeds),
  min_weight_ (1.0),
  do_full_expansion_ (false)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::SequentialSVClustering<PointT>::~SequentialSVClustering ()
{

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::extract (std::map<uint32_t,typename SequentialSV::Ptr > &supervoxel_clusters)
{
  //std::cout << "Init compute  \n";
  bool segmentation_is_possible = initCompute ();
  if ( !segmentation_is_possible )
  {
    deinitCompute ();
    return;
  }

  segmentation_is_possible = prepareForSegmentation ();
  if ( !segmentation_is_possible )
  {
    deinitCompute ();
    return;
  }

  std::vector<size_t> seed_indices;
  selectInitialSupervoxelSeeds (seed_indices);
  createHelpersFromSeedIndices (seed_indices);

  int max_depth = static_cast<int> (sqrt(2)*seed_resolution_/resolution_);
  expandSupervoxels (max_depth);

  makeSupervoxels (supervoxel_clusters);
  deinitCompute ();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::buildVoxelCloud ()
{
  bool segmentation_is_possible = initCompute ();
  if ( !segmentation_is_possible )
  {
    PCL_ERROR ("[pcl::SupervoxelClustering::initCompute] Init failed.\n");
    deinitCompute ();
    return;
  }
  
  //std::cout << "Preparing for segmentation \n";
  segmentation_is_possible = prepareForSegmentation ();
  if ( !segmentation_is_possible )
  {
    PCL_ERROR ("[pcl::SupervoxelClustering::prepareForSegmentation] Building of voxel cloud failed.\n");
    deinitCompute ();
    return;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::extractNewConditionedSupervoxels (SequentialSVMapT &supervoxel_clusters, bool add_new_seeds)
{
  timer_.reset ();
  double t_start = timer_.getTime ();
  std::vector<size_t> existing_seed_indices;
  createHelpersFromWeightMaps (supervoxel_clusters, existing_seed_indices);
  std::cout << "Placing Seeds\n";
  std::vector<size_t> new_seed_indices;
  if (add_new_seeds)
  {
    selectNewSupervoxelSeeds (existing_seed_indices, new_seed_indices);
    std::cout << "Creating helpers - adding "<<new_seed_indices.size ()<<"  new seed indices"<<std::endl;
    appendHelpersFromSeedIndices (new_seed_indices);
  }
  double t_seeds = timer_.getTime ();
  
  clearOwnersSetCentroids ();
  std::cout << "Expanding the supervoxels" << std::endl;
  int max_depth = static_cast<int> (sqrt(2)*seed_resolution_/resolution_);
  expandSupervoxelsFast (max_depth);
  
  
  double t_iterate = timer_.getTime ();
  std::cout << "Making Supervoxel structures" << std::endl;
  makeSupervoxels (supervoxel_clusters);
  double t_supervoxels = timer_.getTime ();
  
  
  std::cout << "--------------------------------- Timing Report --------------------------------- \n";
  std::cout << "Time to seed clusters                          ="<<t_seeds-t_start<<" ms\n";
  std::cout << "Time to expand clusters                        ="<<t_iterate-t_seeds<<" ms\n";
  std::cout << "Time to create supervoxel structures           ="<<t_supervoxels-t_iterate<<" ms\n";
  std::cout << "Total run time                                 ="<<t_supervoxels-t_start<<" ms\n";
  std::cout << "--------------------------------------------------------------------------------- \n";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::expandSupervoxelsFast ( int depth )
{
  for (int i = 0; i < depth; ++i)
  {
    //Expand the the supervoxels one iteration each
    for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
    {
      sv_itr->expand ();
    }
    //Update the centers to reflect new centers
    for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); )
    {
      if (sv_itr->size () == 0)
      {
        sv_itr = supervoxel_helpers_.erase (sv_itr);
      }
      else
      {
        sv_itr->updateCentroid ();
        ++sv_itr;
      } 
    }
    /*
    if ( i < depth - 1 ) //If not on last iteration clear all leaves of ownership 
    {
      typename LeafVectorT::iterator leaf_itr = adjacency_octree_->begin ();
      for (leaf_itr = adjacency_octree_->begin (); leaf_itr != adjacency_octree_->end (); ++leaf_itr)
      {
        VoxelData& voxel = (*leaf_itr)->getData ();
        voxel.owner_ = 0;
        voxel.distance_ = std::numeric_limits<float>::max ();
      }
    }*/
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::makeSupervoxels (SequentialSVMapT &supervoxel_clusters)
{
  supervoxel_clusters.clear ();
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
  {
    uint32_t label = sv_itr->getLabel ();
    std::pair<std::map<uint32_t,typename SequentialSV::Ptr>::iterator,bool> ret;
    ret = supervoxel_clusters.insert (std::pair<uint32_t,typename SequentialSV::Ptr> (label, boost::make_shared<SequentialSV> (label)));
    std::map<uint32_t,typename SequentialSV::Ptr>::iterator new_supervoxel_itr = ret.first;
    sv_itr->getCentroid (new_supervoxel_itr->second->centroid_);
    sv_itr->getVoxels (new_supervoxel_itr->second->voxels_);
  }
  
  initializeLabelColors ();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::createHelpersFromWeightMaps (SequentialSVMapT &supervoxel_clusters, std::vector<size_t> &existing_seed_indices)
{
  existing_seed_indices.clear ();
  supervoxel_helpers_.clear ();
  SequentialSVMapT::iterator sv_itr;
  for  (sv_itr = supervoxel_clusters.begin (); sv_itr != supervoxel_clusters.end (); ++sv_itr )
  {
    supervoxel_helpers_.push_back (new SupervoxelHelper(sv_itr->first,this));
    
    //Go through all indices in the weight map
    std::map <size_t, float>::iterator weight_itr = sv_itr->second->voxel_weight_map_.begin ();
    for ( ; weight_itr != sv_itr->second->voxel_weight_map_.end (); ++weight_itr)
    {
      //Don't bother if below min weight - this prevents expansion into unlabeled regions
      if (weight_itr->second < min_weight_)
        continue;
      //Get the leaf for the index
      LeafContainerT* leaf = adjacency_octree_->at(weight_itr->first);
      VoxelData& voxel = leaf->getData ();
      //Now check if leaf not owned or the weight of this SV owning the leaf is greater than existing 
      //Note that voxel.distance_ here is a weight, not a distance - we're just using the field
      if (voxel.owner_ == 0 || voxel.distance_ < weight_itr->second)
      {
        voxel.owner_ = &(supervoxel_helpers_.back ());
        voxel.distance_ = weight_itr->second;
      }
    }
  }
  
  //Calculate a weighted centroid for each SV
  for  (sv_itr = supervoxel_clusters.begin (); sv_itr != supervoxel_clusters.end (); ++sv_itr )
  {
    //Go through all indices in the weight map
    std::map <size_t, float>::iterator weight_itr = sv_itr->second->voxel_weight_map_.begin ();
    double weight_sum = 0;
    CentroidT centroid;
    centroid.getVector3fMap ().setZero ();
    //Storage for helper update at end
    SupervoxelHelper* helper = 0;
    for ( ; weight_itr != sv_itr->second->voxel_weight_map_.end (); ++weight_itr)
    {
      //Don't bother if below min weight - this prevents expansion into unlabeled regions
      if (weight_itr->second < min_weight_)
        continue;
      //Get the leaf for the index
      LeafContainerT* leaf = adjacency_octree_->at(weight_itr->first);
      VoxelData& voxel = leaf->getData ();
      //If this SV owns this label, add it in
      if (voxel.owner_ != 0 && voxel.owner_->getLabel () == sv_itr->first)
      {
        helper = voxel.owner_;
        centroid.getVector3fMap () += voxel.voxel_centroid_.getVector3fMap () * weight_itr->second;
        weight_sum += weight_itr->second;
        if (!do_full_expansion_)
          helper->addLeaf (leaf);
      }
    }

    //Now set the centroid for the helper
    if (helper != 0)
    {
      centroid.getVector3fMap () /= weight_sum;
      std::vector<int> closest_index;
      std::vector<float> distance;
      voxel_kdtree_->nearestKSearch (centroid, 1, closest_index, distance);
      LeafContainerT* seed_leaf = adjacency_octree_->at (closest_index[0]);
      if (seed_leaf)
      {
        VoxelData& voxel = seed_leaf->getData ();
        if (voxel.owner_ != 0)
          voxel.owner_->removeLeaf (seed_leaf);
        helper->addLeaf (seed_leaf);
        existing_seed_indices.push_back (closest_index[0]);
      }
      else
      {
        PCL_WARN ("Could not find leaf in pcl::SupervoxelClustering<PointT>::createHelpersFromWeightMaps - supervoxel will be deleted \n");
      }
    }
  }
  
  /*
  //TODO This way weights all leaves equally.
  //Now go through and add all leaves to the supervoxel helpers
  //Need to do this after above loop finishes because of weighting
  typename LeafVectorT::iterator leaf_itr = adjacency_octree_->begin ();
  for (leaf_itr = adjacency_octree_->begin (); leaf_itr != adjacency_octree_->end (); ++leaf_itr)
  {
    VoxelData& voxel = (*leaf_itr)->getData ();
    if (voxel.owner_ != 0)
    {
      voxel.owner_->addLeaf (*leaf_itr);
    }
  }
  
  std::vector<int> closest_index;
  std::vector<float> distance;
  existing_seed_indices.reserve (supervoxel_helpers_.size ());
  //Now go through and calculate all centroids based on ownership contained in the supervoxel helpers
  for (typename HelperListT::iterator help_itr = supervoxel_helpers_.begin (); help_itr != supervoxel_helpers_.end (); ++help_itr)
  {
    help_itr->updateCentroid ();
    //Now search for voxel nearest to this new recalculated centroid
    CentroidT centroid;
    help_itr->getCentroid (centroid);
    voxel_kdtree_->nearestKSearch (centroid, 1, closest_index, distance);
    
    //Remove all leaves from the helper 
    //TODO Should we do this? Would mean less expansion - but that would hurt new seeds
    help_itr->removeAllLeaves ();
    LeafContainerT* seed_leaf = adjacency_octree_->at (closest_index[0]);
    if (seed_leaf)
    {
      //TODO Dont need this if we remove all leaves
      VoxelData& voxel = seed_leaf->getData ();
      if (voxel.owner_ != 0)
        voxel.owner_->removeLeaf (seed_leaf);
      help_itr->addLeaf (seed_leaf);
      existing_seed_indices.push_back (closest_index[0]);
    }
    else
    {
      PCL_WARN ("Could not find leaf in pcl::SupervoxelClustering<PointT>::createHelpersFromWeightMaps - supervoxel will be deleted \n");
    }
  }
  */
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::selectNewSupervoxelSeeds (std::vector<size_t> &existing_seed_indices, std::vector<size_t> &seed_indices)
{
  seed_indices.clear ();
  //Initialize octree with voxel centroids - overseed, then prune later
  pcl::octree::OctreePointCloudAdjacency<VoxelT> seed_octree (seed_resolution_/sqrt(2));
  if (use_single_camera_transform_)
    seed_octree.setTransformFunction (boost::bind (&SequentialSVClustering::transformFunctionVoxel, this, _1));  

  seed_octree.setInputCloud (voxel_centroid_cloud_);
  seed_octree.addPointsFromInputCloud ();
  //std::cout << "Size of octree ="<<seed_octree.getLeafCount ()<<"\n";
  std::vector<VoxelT, Eigen::aligned_allocator<VoxelT> > voxel_centers; 
  int num_seeds = seed_octree.getOccupiedVoxelCenters(voxel_centers); 
  std::cout << "Number of seed points before filtering="<<voxel_centers.size ()<<std::endl;

  std::vector<size_t> new_seed_indices;
  new_seed_indices.resize (num_seeds, 0);
  std::vector<int> closest_index (1,0);
  std::vector<float> distance (1,0);

  //Find closest point to seed_resolution octree centroids in voxel_centroid_cloud
  for (int i = 0; i < num_seeds; ++i)  
  {
    if (use_single_camera_transform_)
    {
      //Inverse transform the point.
      voxel_centers[i].z = std::exp (voxel_centers[i].z);
      voxel_centers[i].x *= voxel_centers[i].z;
      voxel_centers[i].y *= voxel_centers[i].z;
    }
    voxel_kdtree_->nearestKSearch (voxel_centers[i], 1, closest_index, distance);
    new_seed_indices[i] = closest_index[0];
  }

  //Shift seeds to voxels within set of neighbors with min curvature (iteratively)
  typename VoxelCloudT::Ptr seed_cloud (new VoxelCloudT);
  seed_cloud->reserve (num_seeds);
  // This is an important parameter - determines maximum shift - here, it is number of voxels per seed
  int search_depth = (seed_resolution_/resolution_) / sqrt (3);
  for (size_t i = 0; i < new_seed_indices.size (); ++i)
  {
    int idx = new_seed_indices[i];
    //Shift based on curvature, number of times based on voxel to seed size ratio
    int new_idx;
    for (int k = 0; k < search_depth; ++k)
    {
      new_idx = findNeighborMinCurvature (idx);
      if (new_idx == idx) //No change - never will be.
        break;
      else
        idx = new_idx;
    }
    new_seed_indices[i] = idx;
    seed_cloud->push_back (voxel_centroid_cloud_->points[idx]);
  }

  //Create separate cloud for existing seeds
  typename VoxelCloudT::Ptr existing_seed_cloud (new VoxelCloudT);
  existing_seed_cloud->reserve (existing_seed_indices.size ());
  for (size_t i = 0; i < existing_seed_indices.size (); ++i)
  {
    existing_seed_cloud->push_back (voxel_centroid_cloud_->points[existing_seed_indices[i]]);
  }

  //Build Kdtree for seed cloud for pruning
  typename pcl::search::KdTree<VoxelT> seed_kdtree;
  seed_kdtree.setInputCloud (seed_cloud);
  //Build Kdtree for existing seed cloud for pruning
  typename pcl::search::KdTree<VoxelT> existing_seed_kdtree;
  existing_seed_kdtree.setInputCloud (existing_seed_cloud);
  
  std::vector<int> neighbors;
  std::vector<float> sqr_distances;
  //This stores seed_cloud index to SeedNHood pointers
  std::vector<typename SeedNHood::Ptr> seed_lookup (seed_cloud->size ());
  //Now we check if seeds are near to other seeds, and prune those that are 
  //We do this by pruning seeds with the most edges first
  //We consider existing supervoxels for edge numbers, but they cannot be pruned.
  typename SeedNHood::SeedPriorityQueue seed_heap;
  std::vector<typename SeedNHood::Ptr> seed_nhoods;
  //float search_radius = seed_resolution_/ 2;
  float angle_thresh_rad = (0.785398163); // PI/4
  for (size_t i = 0; i < new_seed_indices.size (); ++i)  
  {
    size_t voxel_idx = new_seed_indices[i];
    //Search radius needs to be adjusted depending on transform and voxel coordinates.
    float search_radius = seed_prune_radius_;
    if ( use_single_camera_transform_ )
    {
      float dist_from_origin = voxel_centroid_cloud_->at(voxel_idx).getVector3fMap ().squaredNorm ();
      search_radius *= std::log (dist_from_origin + 2.71828); //0 dist has radius*=1
    }

    seed_kdtree.radiusSearch (voxel_centroid_cloud_->at(voxel_idx), search_radius , neighbors, sqr_distances);
    typename SeedNHood::Ptr new_nhood (new SeedNHood);
    seed_lookup[i] = new_nhood;
    new_nhood->neighbor_indices_.reserve (neighbors.size ());
    //check each neighbor for normal difference - if > angle_thresh, don't count it as an edge
    for (int k = 0; k < neighbors.size (); ++k)
    {
      float angle_diff_rad = std::acos(seed_cloud->at(i).getNormalVector3fMap ().dot (seed_cloud->at(neighbors[k]).getNormalVector3fMap ()));
      if (std::isnan(angle_diff_rad) || angle_diff_rad < angle_thresh_rad)
      {
        new_nhood->neighbor_indices_.push_back (neighbors[k]);
      }
    }
    new_nhood->voxel_idx_ = voxel_idx;
    new_nhood->seed_idx_ = i;
    new_nhood->num_active_ = new_nhood->neighbor_indices_.size ();

    //Now search the existing seeds, add as edges
    existing_seed_kdtree.radiusSearch (voxel_centroid_cloud_->at(voxel_idx), search_radius , neighbors, sqr_distances);
    new_nhood->num_active_ += neighbors.size ();

    //Push the NHOOD onto the heap
    new_nhood->handle_ = seed_heap.push (new_nhood);
  }

  int max_in_radius = 1;
  int num_removed = 0;
  while (seed_heap.size () > 0 && seed_heap.top ()->num_active_ > max_in_radius)
  {
    int idx_to_remove = seed_heap.top ()->seed_idx_;
    typename SeedNHood::Ptr seed_to_remove = seed_lookup[idx_to_remove];
    seed_heap.pop ();
    seed_to_remove->num_active_ = 0;
    //Go through neighbor indices, lookup, decriment
    for (int i = 0; i < seed_to_remove->neighbor_indices_.size (); ++i)
    {
      typename SeedNHood::Ptr temp = seed_lookup[seed_to_remove->neighbor_indices_[i]];
      if (temp->num_active_ > 0)
      {
        temp->num_active_--;
        seed_heap.decrease (temp->handle_);
      }
    }
    ++num_removed;
  }
  
  //Now clear seed indices and push remaining seeds onto it
  seed_indices.clear ();
  seed_indices.reserve (seed_heap.size ());
  for (typename SeedNHood::SeedPriorityQueue::const_iterator itr = seed_heap.begin (); itr != seed_heap.end (); ++itr)
  {
    seed_indices.push_back ((*itr)->voxel_idx_);
  }
  std::cout <<"Removed "<<num_removed<<" seeds, seed points after filtering="<<seed_indices.size ()<<std::endl;
  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::clearOwnersSetCentroids ()
{
  int num_erased = 0;
  //Update centroids and delete empties
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); )
  {
    if (sv_itr->size () == 0)
    {
      sv_itr = supervoxel_helpers_.erase (sv_itr);
      ++num_erased;
    }
    else
    {
      sv_itr->updateCentroid ();
      ++sv_itr;
    } 
  }
  std::cout <<"Erased "<<num_erased<<" Supervoxels!\n";
  //Clear all leaves of ownership 
  typename LeafVectorT::iterator leaf_itr = adjacency_octree_->begin ();
  for (leaf_itr = adjacency_octree_->begin (); leaf_itr != adjacency_octree_->end (); ++leaf_itr)
  {
    VoxelData& voxel = (*leaf_itr)->getData ();
    voxel.owner_ = 0;
    voxel.distance_ = std::numeric_limits<float>::max ();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::appendHelpersFromSeedIndices (std::vector<size_t> &seed_indices)
{
  //TODO REMOVE
  int num_helpers_before = supervoxel_helpers_.size ();
  int max_label_before = SupervoxelClustering<PointT>::getMaxLabel ();
  //////////////////////////////////
  int num_not_added = 0;
  uint32_t next_label = 1;
  typename HelperListT::iterator help_itr = supervoxel_helpers_.begin ();
  if (help_itr != supervoxel_helpers_.end ())
    ++help_itr;
  typename HelperListT::iterator help_itr2 = supervoxel_helpers_.begin ();
  for (size_t i = 0; i < seed_indices.size (); ++i)
  {
    //First check if the leaf is already owned - if so, don't add a new seed here.
    LeafContainerT* seed_leaf = adjacency_octree_->at(seed_indices[i]);//adjacency_octree_->getLeafContainerAtPoint 
    if (seed_leaf)
    {
      VoxelData& voxel = seed_leaf->getData ();
      if (voxel.owner_ != 0)
      {
        ++num_not_added;
        continue;
      }
    }else
    {
      PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::appendHelpersFromSeedIndices - supervoxel will be deleted \n");
      continue;
    }
    //If we want to add this, need to find a free label
    //The first iterator always stays one ahead - when we find a gap in labels we break
    while (help_itr != supervoxel_helpers_.end () && help_itr->getLabel () == help_itr2->getLabel () + 1)
    {
      ++help_itr; ++help_itr2;
    }
    
    next_label = help_itr2->getLabel () + 1;
    supervoxel_helpers_.insert (help_itr, new SupervoxelHelper(next_label,this));
    // this makes itr2 point to the new element
    ++help_itr2;
    help_itr2->addLeaf (seed_leaf);
  }

  std::cout <<"Did not add "<<num_not_added<<" due to already owned!\n";
  //TODO Can remove this - just for debugging
  int num_helpers_after = supervoxel_helpers_.size ();
  int max_label_after = SupervoxelClustering<PointT>::getMaxLabel ();
  std::cout <<"Num SV Before ="<<num_helpers_before<<"  maxlabel="<<max_label_before<<"     Num SV After ="<<num_helpers_after<<"  maxlabel="<<max_label_after<<"   num new seeds="<<seed_indices.size ()<<"\n";
  int empty_slots = max_label_before - num_helpers_before;
  //Either the max label hasn't changed (we only filled in holes), or the max label has increased and there are no holes
  assert ( (max_label_before == max_label_after) || (max_label_after  == (max_label_before + seed_indices.size () - empty_slots)));
}

#define PCL_INSTANTIATE_SequentialSVClustering(T) template class PCL_EXPORTS pcl::SequentialSVClustering<T>;

#endif    // PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_
 

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
  SupervoxelClustering<PointT> (voxel_resolution, seed_resolution, use_single_camera_transform, prune_close_seeds)
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

  std::vector<int> seed_indices;
  selectInitialSupervoxelSeeds (seed_indices);
  createHelpersFromSeedIndices (seed_indices);

  int max_depth = static_cast<int> (sqrt(3)*seed_resolution_/resolution_);
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
pcl::SequentialSVClustering<PointT>::extractNewConditionedSupervoxels (SequentialSVMapT &supervoxel_clusters)
{
  initializeLabelColors ();
  timer_.reset ();
  double t_start = timer_.getTime ();
  std::vector<size_t> updated_seed_indices;
  createHelpersFromWeightMaps (supervoxel_clusters, updated_seed_indices);
  std::cout << "Placing Seeds" << std::endl;
  std::vector<int> seed_indices;
  //selectInitialSupervoxelSeeds (seed_indices);
  std::cout << "Creating helpers "<<std::endl;
  //createSupervoxelHelpers (seed_indices);
  double t_seeds = timer_.getTime ();
  
  std::cout << "Expanding the supervoxels" << std::endl;
  int max_depth = static_cast<int> (sqrt(3)*seed_resolution_/resolution_);
  //expandSupervoxels (max_depth);
  
  double t_iterate = timer_.getTime ();
  std::cout << "Making Supervoxel structures" << std::endl;
  //makeSupervoxels (supervoxel_clusters);
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

template <typename PointT> void
pcl::SequentialSVClustering<PointT>::createHelpersFromWeightMaps (SequentialSVMapT &supervoxel_clusters, std::vector<size_t> &updated_seed_indices)
{
  std::cout <<"Creating helpers\n";
  updated_seed_indices.clear ();
  supervoxel_helpers_.clear ();
  SequentialSVMapT::iterator sv_itr;
  for  (sv_itr = supervoxel_clusters.begin (); sv_itr != supervoxel_clusters.end (); ++sv_itr )
  {
    supervoxel_helpers_.push_back (new SupervoxelHelper(sv_itr->first,this));
    
    //Go through all indices in the weight map
    std::map <size_t, float>::iterator weight_itr = sv_itr->second->voxel_weight_map_.begin ();
    for ( ; weight_itr != sv_itr->second->voxel_weight_map_.end (); ++weight_itr)
    {
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
  std::cout <<"Adding leaves\n";
  
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
  updated_seed_indices.reserve (supervoxel_helpers_.size ());
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
    //help_itr->removeAllLeaves ();
    LeafContainerT* seed_leaf = adjacency_octree_->at (closest_index[0]);
    if (seed_leaf)
    {
      //TODO Dont need this if we remove all leaves
      VoxelData& voxel = seed_leaf->getData ();
      if (voxel.owner_ != 0)
        voxel.owner_->removeLeaf (seed_leaf);
      help_itr->addLeaf (seed_leaf);
      updated_seed_indices.push_back (closest_index[0]);
    }
    else
    {
      PCL_WARN ("Could not find leaf in pcl::SupervoxelClustering<PointT>::createHelpersFromWeightMaps - supervoxel will be deleted \n");
    }
  }
  
  //Clear all leaves of ownership 
  for (leaf_itr = adjacency_octree_->begin (); leaf_itr != adjacency_octree_->end (); ++leaf_itr)
  {
    VoxelData& voxel = (*leaf_itr)->getData ();
    voxel.owner_ = 0;
    voxel.distance_ = std::numeric_limits<float>::max ();
  }

}

#define PCL_INSTANTIATE_SequentialSVClustering(T) template class PCL_EXPORTS pcl::SequentialSVClustering<T>;

#endif    // PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_
 

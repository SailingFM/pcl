/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012, Jeremie Papon
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 */

#ifndef PCL_OCTREE_POINTCLOUD_SEQUENTIAL_HPP_
#define PCL_OCTREE_POINTCLOUD_SEQUENTIAL_HPP_

#include <pcl/octree/octree_pointcloud_sequential.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> 
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::OctreePointCloudSequential (const double resolution_arg) 
: OctreePointCloud<PointT, LeafContainerT, BranchContainerT> (resolution_arg),
  diff_func_ (0),
  difference_threshold_ (0.1f),
  occlusion_test_interval_ (0.5f),
  threads_ (1),
  stored_keys_valid_ (false)
{
 
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::addPointsFromInputCloud ()
{
  //If we're empty, just call the adjacency version 
  //if (leaf_key_vec_.size () == 0)
  //{
  //  OctreePointCloudAdjacency<PointT, LeafContainerT, BranchContainerT>::addPointsFromInputCloud ();
  //  //Initialize the old point value in data containers - this is needed because we swap them
  //  typename LeafKeyVectorT::iterator pair_itr;
  //  for (pair_itr = leaf_key_vec_.begin () ; pair_itr != leaf_key_vec_.end (); ++pair_itr)
  //  {
  //    (pair_itr->first)->getData ().initLastPoint ();
  //  }
  //  return;
  //}
  //Otherwise, start sequential update
  //Go through and reset all the containers for the new frame.
  typename LeafKeyVectorT::iterator pair_itr;
  for (pair_itr = leaf_key_vec_.begin () ; pair_itr != leaf_key_vec_.end (); ++pair_itr)
  {
    (pair_itr->first)->getData ().prepareForNewFrame ((pair_itr->first)->getPointCounter ());
    (pair_itr->first)->resetPointCount ();
  }
  
  LeafContainerT *leaf_container;
  new_frame_pairs_.clear ();
  new_frame_pairs_.reserve (this->getLeafCount ());
  
  //Adapt the octree to fit all input points
  int depth_before_addition = this->octree_depth_;
  for (size_t i = 0; i < input_->points.size (); i++)
  {
    this->adoptBoundingBoxtoPoint (input_->points[i]);
  }
  //If depth has changed all stored keys are not valid
  if (this->octree_depth_ != depth_before_addition)
  {
    stored_keys_valid_ = false;
  }
   
  //Now go through and add all points to the octree- keys are already adapted
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1024) num_threads(threads_)
  #endif
  for (size_t i = 0; i < input_->points.size (); i++)
  {
    // add the point to octree
    addPointSequential (static_cast<unsigned int> (i));
  }
  
  //If Geometrically new - need to reciprocally update neighbors and compute Data
  //New frame leaf/key vectors now contain only the geometric new leaves
  for (size_t i = 0; i < new_frame_pairs_.size (); ++i)
  {
      
    computeNeighbors (new_frame_pairs_[i]);
    (new_frame_pairs_[i].first)->computeData ();
    (new_frame_pairs_[i].first)->getData ().initLastPoint ();
    //(new_frame_pairs_[i].first)->getData ().setNew (true);
  }
  
  // This will store old leaves that will need to be deleted
  LeafKeyVectorT delete_list;
  delete_list.reserve (this->getLeafCount () / 4); //Probably a reasonable upper bound
  
  //Now we need to iterate through all leaves from previous frame - 
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1024) shared (delete_list) private (leaf_container) num_threads(threads_)
  #endif
  for (size_t i = 0; i < leaf_key_vec_.size (); ++i)
  {
    leaf_container = leaf_key_vec_[i].first;
    //If the stored keys aren't valid due to bounding box changing, update them
    if (!stored_keys_valid_)
    {
      this->genOctreeKeyforPoint (leaf_container->getData ()->getPoint (), *leaf_key_vec_[i].second);
    }
    //If no neighbors probably noise - delete 
    if (leaf_container->getNumNeighbors () == 1)
    {
      #ifdef _OPENMP
      #pragma omp critical (delete_list)
      #endif
      {
        delete_list.push_back (leaf_key_vec_[i]); 
      }
    }
    //Check if the leaf had no points observed this frame
    else if (leaf_container->getPointCounter () == 0)
    {
      float voxels_to_occluder = testForOcclusion (leaf_key_vec_[i]);
      //If occluded (distance to occluder != 0)
      if (voxels_to_occluder != 0.0f)
      {
        //If occluder is right next to it, and it has new neighbor, it can be removed
        //This is basically a test to remove extra voxels caused by objects moving towards the camera
        if ( voxels_to_occluder <= 1.0f && testForNewNeighbors (leaf_container))
        {
          #ifdef _OPENMP
          #pragma omp critical (delete_list)
          #endif
          {  
            delete_list.push_back (leaf_key_vec_[i]); 
          }
        }
        else //otherwise add it to the current leaves and revert it to last timestep (since current has nothing in it) 
        { //TODO Maybe maintain a separate list of occluded leaves?
          #ifdef _OPENMP
          #pragma omp critical (new_frame)
          #endif
          {   
            new_frame_pairs_.push_back (leaf_key_vec_[i]);  
          }
          leaf_container->getData ().revertToLastPoint ();
        }
      }
      else //not occluded & not observed safe to delete
      { 
        #ifdef _OPENMP
        #pragma omp critical (delete_list)
        #endif
        {  
          delete_list.push_back (leaf_key_vec_[i]);
        }
      }
    }
    else //Existed in previous frame and observed so just update data
    {
      #ifdef _OPENMP
      #pragma omp critical (new_frame)
      #endif
      {  
        new_frame_pairs_.push_back (leaf_key_vec_[i]);
      }
      //Compute the data from the points added to the voxel container
      leaf_container->computeData ();
      //Use the difference function to check if the leaf has changed
      if ( diff_func_ && diff_func_ (leaf_container) > difference_threshold_)
      {
        leaf_container->getData ().setChanged (true);
      } 
    }
    
  }

    
  //Swap new leaf_key vector (which now contains old and new combined) for old (which is not needed anymore)
  leaf_key_vec_.swap (new_frame_pairs_);
  
  //All keys which were stored in new_frame_pairs_ are valid, so this is now true for leaf_key_vec_
  stored_keys_valid_ = true;
  
  //Go through and delete voxels scheduled
  for (typename LeafKeyVectorT::iterator delete_itr = delete_list.begin (); delete_itr != delete_list.end (); ++delete_itr)
  {
    leaf_container = delete_itr->first;
    //Remove pointer to it from all neighbors
    typename std::set<LeafContainerT*>::iterator neighbor_itr = leaf_container->begin ();
    typename std::set<LeafContainerT*>::iterator neighbor_end = leaf_container->end ();
    for ( ; neighbor_itr != neighbor_end; ++neighbor_itr)
    {
      (*neighbor_itr)->removeNeighbor (leaf_container);
    }
    //Now delete the leaf - there is a problem with this function, sometimes (rarely) it doesn't delete
    this->removeLeaf ( *(delete_itr->second) );
  }
  
  //Final check to make sure they match the leaf_key_vector is correct size after deletion
  assert (leaf_key_vec_.size () == this->getLeafCount ());
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::addPointSequential (const int point_index_arg)
{
  const PointT& point = this->input_->points[point_index_arg];
  
  if (!isFinite (point))
    return;
  
  boost::shared_ptr<OctreeKey> key (new OctreeKey);
  
  // generate key - use adjacency function since it possibly has a transform
  this->OctreePointCloudT::genOctreeKeyforPoint (point, *key);
  
  // Check if leaf exists in octree at key
  LeafContainerT* container = this->findLeaf(*key);
  
  if (container == 0) //If not, do a lock and add the leaf
  {
    boost::mutex::scoped_lock (create_mutex_);
    //Check again, since another thread might have created between the first find and now
    container = this->findLeaf(*key);
    if (container == 0)
    {
      container = this->createLeaf(*key); //This is fine if the leaf has already been created by another
      if (container == 0)
      {
        PCL_ERROR ("FAILED TO CREATE LEAF in OctreePointCloudSequential::addPointSequential");
        return;
      }
      new_frame_pairs_.push_back (std::make_pair (container, key));
    }
  }
  //Add the point to the leaf
  container->addPoint (point);
  
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::computeNeighbors (LeafKeyPairT& leaf_key_arg)
{ 
  //Make sure requested key is valid
  if (leaf_key_arg.second->x > this->max_key_.x || leaf_key_arg.second->y > this->max_key_.y || leaf_key_arg.second->z > this->max_key_.z)
  {
    PCL_ERROR ("OctreePointCloudAdjacency::computeNeighbors Requested neighbors for invalid octree key\n");
    return;
  }
  
  OctreeKey neighbor_key;
  int dx_min = (leaf_key_arg.second->x > 0) ? -1 : 0;
  int dy_min = (leaf_key_arg.second->y > 0) ? -1 : 0;
  int dz_min = (leaf_key_arg.second->z > 0) ? -1 : 0;
  int dx_max = (leaf_key_arg.second->x == this->max_key_.x) ? 0 : 1;
  int dy_max = (leaf_key_arg.second->y == this->max_key_.y) ? 0 : 1;
  int dz_max = (leaf_key_arg.second->z == this->max_key_.z) ? 0 : 1;
  
  for (int dx = dx_min; dx <= dx_max; ++dx)
  {
    for (int dy = dy_min; dy <= dy_max; ++dy)
    {
      for (int dz = dz_min; dz <= dz_max; ++dz)
      {
        neighbor_key.x = static_cast<uint32_t> (leaf_key_arg.second->x + dx);
        neighbor_key.y = static_cast<uint32_t> (leaf_key_arg.second->y + dy);
        neighbor_key.z = static_cast<uint32_t> (leaf_key_arg.second->z + dz);
        LeafContainerT *neighbor = this->findLeaf (neighbor_key);
        if (neighbor)
        {
          leaf_key_arg.first->addNeighbor (neighbor);
          neighbor->addNeighbor (leaf_key_arg.first);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> bool
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::testForNewNeighbors (const LeafContainerT* leaf_container) const
{
  typename LeafContainerT::const_iterator neighb_itr = leaf_container->begin ();
  for ( ; neighb_itr != leaf_container->end (); ++neighb_itr)
  {
    if ( (*neighb_itr)->getData ().isNew () )
      return true;
  }
  return false;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> float
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::testForOcclusion (const LeafKeyPairT &leaf_key_pair , const OctreeKey& camera_key) const
{
  
  OctreeKey current_key = *(leaf_key_pair.second);
  
  Eigen::Vector3f camera_key_vals (camera_key.x, camera_key.y, camera_key.z);
  Eigen::Vector3f leaf_key_vals (current_key.x,current_key.y,current_key.z);
  Eigen::Vector3f direction = (camera_key_vals - leaf_key_vals);
  float norm = direction.norm ();
  direction.normalize ();
  
  const int nsteps = std::max (1, static_cast<int> (norm / occlusion_test_interval_));
  leaf_key_vals += (direction * occlusion_test_interval_);
  OctreeKey test_key;
  
  // Walk along the line segment with small steps.
  for (int i = 1; i < nsteps; ++i)
  {
    //Start at the leaf voxel, and move back towards sensor.
    leaf_key_vals += (direction * occlusion_test_interval_);
    //This is a shortcut check - if we're outside of the bounding box of the 
    //octree there's no possible occluders. It might be worth it to check all, but < min_z_ is probably sufficient.
    if (leaf_key_vals.z () <= 0) 
      return false;
    //Now we need to round the key
      test_key.x = ::round(leaf_key_vals.x ());
      test_key.y = ::round(leaf_key_vals.y ());
      test_key.z = ::round(leaf_key_vals.z ());
      
      if (test_key == current_key)
        continue;
      
      current_key = test_key;
      
      //If the voxel is occupied, there is a possible occlusion
      if (this->findLeaf (test_key))
      {
        float voxels_to_occluder = i * occlusion_test_interval_;
        return voxels_to_occluder; 
      }
  }
  //If we didn't run into a leaf on the way to this camera, it can't be occluded.
  return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> typename pcl::PointCloud<PointT>::Ptr
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::getNewVoxelCloud ()
{
  typename PointCloudT::Ptr new_cloud (new PointCloudT ());
  new_cloud->reserve (leaf_key_vec_.size ());
  typename LeafKeyVectorT::iterator leaf_itr;
  for (leaf_itr = leaf_key_vec_.begin () ; leaf_itr != leaf_key_vec_.end (); ++leaf_itr)
  {
    if ( (leaf_itr->first)->getData ().isNew () || (leaf_itr->first)->getData ().isChanged ())
    {
      new_cloud->push_back ( (leaf_itr->first)->getData ().getPoint ());
    }
  }
  return new_cloud;
}


////////////////////////////////////////////////////////////////////////////////
// The rest are container explicit instantiations for XYZ, XYZRGB, and XYZRGBA  point types ///
////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  namespace octree
  {
    /// XYZRGBA  ////////////////////////////////////////
    template<>
    float
    OctreePointCloudSequential<PointXYZRGBA,
    OctreePointCloudAdjacencyContainer <PointXYZRGBA, SequentialVoxelData<PointXYZRGBA> > > 
    ::SeqVoxelDataDiff (const OctreePointCloudAdjacencyContainer <PointXYZRGBA, SequentialVoxelData<PointXYZRGBA> >* leaf)
    {
      float temp1 = leaf->getData ().rgb_.norm () ;
      float temp2 = leaf->getData ().rgb_old_.norm ();
      
      return 1.0f - (leaf->getData ().rgb_ / temp1).dot ((leaf->getData ().rgb_old_ / temp2));
    }
    
    template<>
    void
    OctreePointCloudAdjacencyContainer<PointXYZRGBA,
    SequentialVoxelData<PointXYZRGBA> >::addPoint (const pcl::PointXYZRGBA &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.num_points_++;     
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[0] += static_cast<float> (new_point.r); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[1] += static_cast<float> (new_point.g); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[2] += static_cast<float> (new_point.b); 
    }
    
    template<> void
    OctreePointCloudAdjacencyContainer<PointXYZRGBA,
    SequentialVoxelData<PointXYZRGBA> >::computeData ()
    {
      data_.rgb_ /= static_cast<float> (data_.num_points_);
      data_.xyz_ /= static_cast<float> (data_.num_points_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZRGBA>::getPoint (pcl::PointXYZRGBA &point_arg ) const
    {
      point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 | 
      static_cast<uint32_t>(rgb_[1]) << 8 | 
      static_cast<uint32_t>(rgb_[2]);  
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
    
    // XYZRGB ///////////////////////////////
    template<>
    float
    OctreePointCloudSequential<PointXYZRGB,
    OctreePointCloudAdjacencyContainer <PointXYZRGB, SequentialVoxelData<PointXYZRGB> > > 
    ::SeqVoxelDataDiff (const OctreePointCloudAdjacencyContainer <PointXYZRGB, SequentialVoxelData<PointXYZRGB> >* leaf)
    {
      return (leaf->getData ().rgb_ - leaf->getData ().rgb_old_).norm () / 255.0f;
    }
    
    template<>
    void
    OctreePointCloudAdjacencyContainer<PointXYZRGB,
    SequentialVoxelData<PointXYZRGB> >::addPoint (const pcl::PointXYZRGB &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++data_.num_points_;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[0] += static_cast<float> (new_point.r); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[1] += static_cast<float> (new_point.g); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[2] += static_cast<float> (new_point.b); 
    }
    
    template<> void
    OctreePointCloudAdjacencyContainer<PointXYZRGB,
    SequentialVoxelData<PointXYZRGB> >::computeData ()
    {
      data_.rgb_ /= static_cast<float> (data_.num_points_);
      data_.xyz_ /= static_cast<float> (data_.num_points_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZRGB>::getPoint (pcl::PointXYZRGB &point_arg ) const
    {
      // In XYZRGB you need to do this nonsense
      uint32_t temp_rgb = static_cast<uint32_t>(rgb_[0]) << 16 | 
      static_cast<uint32_t>(rgb_[1]) << 8 | 
      static_cast<uint32_t>(rgb_[2]);
      point_arg.rgb = *reinterpret_cast<float*> (&temp_rgb);  
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
    
    // XYZ /////////////////////////////////////////
    template<>
    void
    OctreePointCloudAdjacencyContainer<PointXYZ,
    SequentialVoxelData<PointXYZ> >::addPoint (const pcl::PointXYZ &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++data_.num_points_;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
    }
    
    template<> void
    OctreePointCloudAdjacencyContainer<PointXYZ,
    SequentialVoxelData<PointXYZ> >::computeData ()
    {
      data_.xyz_ /= static_cast<float> (num_points_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZ>::getPoint (pcl::PointXYZ &point_arg ) const
    {
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
  }
}

#define PCL_INSTANTIATE_OctreePointCloudSequential(T) template class PCL_EXPORTS pcl::octree::OctreePointCloudSequential<T>;

#endif

